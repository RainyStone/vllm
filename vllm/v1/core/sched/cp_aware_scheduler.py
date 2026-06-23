# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CP-Aware Scheduler for Dynamic Context Parallel.

Extends the existing Scheduler with CP awareness while preserving the
per-DP independent process architecture. CP requests are coordinated
via a post-schedule all-reduce MIN consensus protocol.

Design rationale: the client layer (DPEngineCoreClient) broadcasts CP requests
to ALL DP ranks before they reach the scheduler, so every rank is guaranteed to
hold the same CP request. No pre-schedule announce/vote phase is needed.
Coordination is a single post-schedule all-reduce MIN that agrees on whether
each rank successfully scheduled each CP request in the current step.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = init_logger(__name__)

# Maximum number of CP requests that can be synced in one round.
_MAX_CP_SYNC_SLOTS = 32

# Three-state encoding for post-schedule consensus.
SCHEDULED = 2
NOT_SCHEDULED = 1
PREEMPTED = 0


class CPSyncProtocol:
    """Post-schedule consensus protocol for CP request scheduling.

    Uses the existing dp_group all-reduce MIN to agree on whether each active
    CP request was successfully scheduled on ALL ranks in the current step.

    Three-state encoding per request:
        SCHEDULED (2)     - this rank scheduled the request
        NOT_SCHEDULED (1) - this rank did not schedule it (budget exhausted)
        PREEMPTED (0)     - this rank preempted it (KV cache eviction)

    After all-reduce MIN:
        min >= SCHEDULED  -> confirmed: execute on all ranks
        min >= NOT_SCHEDULED -> soft rollback: re-queue without resetting KV
        min == PREEMPTED  -> hard rollback: all ranks evict and reset
    """

    def __init__(
        self,
        dp_group: "ProcessGroup",
        cp_world_size: int,
        cp_rank: int,
    ):
        self.dp_group = dp_group
        self.cp_world_size = cp_world_size
        self.cp_rank = cp_rank

        # Pre-allocated tensor for post-schedule all-reduce (avoids per-call
        # allocation on the hot path).
        self._confirm_tensor = torch.zeros(
            _MAX_CP_SYNC_SLOTS, dtype=torch.int32, device="cpu"
        )

    def sync_schedule_confirm(
        self,
        active_ids: list[str],
        status: list[int],
    ) -> tuple[list[str], list[str], list[str]]:
        """Post-schedule consensus using three-state encoding.

        Args:
            active_ids: Active CP request IDs (sorted, identical across ranks).
            status: Per-request status on this rank
                    (SCHEDULED=2 / NOT_SCHEDULED=1 / PREEMPTED=0).

        Returns:
            (confirmed_ids, soft_rollback_ids, hard_rollback_ids)
        """
        num_slots = min(len(active_ids), _MAX_CP_SYNC_SLOTS)
        if num_slots == 0:
            return [], [], []

        self._confirm_tensor.zero_()
        for i in range(num_slots):
            self._confirm_tensor[i] = status[i]

        torch.distributed.all_reduce(
            self._confirm_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.dp_group,
        )

        confirmed: list[str] = []
        soft_rollback: list[str] = []
        hard_rollback: list[str] = []
        for i in range(num_slots):
            val = self._confirm_tensor[i].item()
            if val >= SCHEDULED:
                confirmed.append(active_ids[i])
            elif val >= NOT_SCHEDULED:
                soft_rollback.append(active_ids[i])
            else:
                hard_rollback.append(active_ids[i])

        if soft_rollback or hard_rollback:
            logger.debug(
                "CP confirm: confirmed=%d soft_rollback=%d hard_rollback=%d"
                " on rank %d",
                len(confirmed),
                len(soft_rollback),
                len(hard_rollback),
                self.cp_rank,
            )

        return confirmed, soft_rollback, hard_rollback

    def sync_empty(self) -> None:
        """Participate in the sync all_reduce with no active CP requests.

        Called by ranks that have no active CP requests in a given step so
        that peer ranks which do have active requests are not blocked waiting
        for all participants in the collective operation.

        Fills the tensor with NOT_SCHEDULED (1) so that the MIN operation does
        not drive any active slot down to PREEMPTED (0), which would trigger
        spurious hard-rollbacks on the peer ranks.
        """
        self._confirm_tensor.fill_(NOT_SCHEDULED)
        torch.distributed.all_reduce(
            self._confirm_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.dp_group,
        )


class CPAwareScheduler(Scheduler):
    """Scheduler with CP awareness for distributed DYCP.

    Each DP rank runs its own CPAwareScheduler. CP requests are coordinated
    via a distributed sync protocol using the existing dp_group.

    Key differences from base Scheduler:
    - Classifies requests as long (CP) or short (DP) based on token threshold
    - CP requests are activated immediately on arrival (client guarantees
      broadcast to all ranks, so no announce phase is needed)
    - Only allocates local portion of KV cache for CP requests
    - Adds CP metadata to SchedulerOutput
    - Post-schedule all-reduce MIN agrees on confirmed/rollback decisions
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        dp_group: "ProcessGroup | None" = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        logger.info("Using CPAwareScheduler.")

        self.cp_world_size = vllm_config.parallel_config.dycp_size
        self.cp_rank = (
            vllm_config.parallel_config.data_parallel_rank % self.cp_world_size
        )
        self.long_request_threshold = (
            vllm_config.scheduler_config.long_request_threshold
        )
        self.max_cp_requests = vllm_config.scheduler_config.num_cp_seqs

        # Active CP requests: request_id -> Request.
        # CP requests are added here immediately on arrival; no pending state.
        self.active_cp_requests: dict[str, Request] = {}

        # Tracks CP requests preempted by schedule() in the current step.
        self._preempted_this_step: set[str] = set()

        # CP sync protocol (None if dp_group not provided, e.g., single DP).
        self.cp_sync: CPSyncProtocol | None = None
        if dp_group is not None and self.cp_world_size > 1:
            self.cp_sync = CPSyncProtocol(
                dp_group=dp_group,
                cp_world_size=self.cp_world_size,
                cp_rank=self.cp_rank,
            )

    # ------------------------------------------------------------------
    # Request classification and routing
    # ------------------------------------------------------------------

    def _is_long_request(self, request: Request) -> bool:
        """Classify request as long (CP) based on prefill token count."""
        num_prefill_tokens = request.num_tokens - request.num_output_tokens
        return num_prefill_tokens >= self.long_request_threshold

    def _get_local_cp_tokens(self, total_tokens: int) -> int:
        """Calculate how many tokens this rank stores for a CP request."""
        base = total_tokens // self.cp_world_size
        remainder = total_tokens % self.cp_world_size
        return base + (1 if self.cp_rank < remainder else 0)

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Add request, routing CP requests directly to active state."""
        if self.cp_world_size <= 1 or not self._is_long_request(request):
            request.cp_ranks = [self.cp_rank]
            super().add_request(request)
        else:
            # Client guarantees CP requests are broadcast to all ranks, so
            # activate immediately without a pending/announce phase.
            logger.debug("[DYCP Debug] Long request detected.")
            request.cp_ranks = list(range(self.cp_world_size))
            self.active_cp_requests[request.request_id] = request
            self.requests[request.request_id] = request
            self.waiting.add_request(request)
            logger.debug(
                "CP request %s activated on rank %d (tokens=%d)",
                request.request_id,
                self.cp_rank,
                request.num_tokens,
            )

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Override to track CP requests preempted by schedule()."""
        if request.request_id in self.active_cp_requests:
            self._preempted_this_step.add(request.request_id)
        super()._preempt_request(request, timestamp)

    # ------------------------------------------------------------------
    # has_requests override
    # ------------------------------------------------------------------

    def has_requests(self) -> bool:
        # Active CP requests require this rank to participate in the all_reduce
        # even when it has no locally-scheduled tokens, so treat them as
        # "requests to handle" from core.py's perspective.
        return super().has_requests() or bool(self.active_cp_requests)

    # ------------------------------------------------------------------
    # Schedule override: local scheduling + distributed CP consensus
    # ------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        """Schedule requests and run the distributed CP consensus in one step.

        When cp_sync is active this method performs:
          1. super().schedule() — local scheduling as usual
          2. Annotate output with CP metadata
          3. all_reduce MIN — agree on confirmed / rollback decisions
          4. Apply rollbacks and finalise cp_req_ids_sorted
        """
        self._preempted_this_step.clear()
        output = super().schedule()

        # Annotate output with CP metadata (needed by workers regardless of sync).
        output.cp_rank = self.cp_rank
        cp_req_ids: list[str] = []
        req_id_to_cp_size: dict[str, list[int]] = {}
        for req_id in output.num_scheduled_tokens:
            if req_id in self.active_cp_requests:
                cp_req_ids.append(req_id)
                req_id_to_cp_size[req_id] = self.active_cp_requests[req_id].cp_ranks
        output.num_cp_request = len(cp_req_ids)
        output.cp_rank_to_req_id = cp_req_ids if cp_req_ids else None
        output.req_id_to_cp_size = req_id_to_cp_size if req_id_to_cp_size else None
        if output.cp_rank_scheduled_tokens is None:
            output.cp_rank_scheduled_tokens = {}
        for req_id in cp_req_ids:
            output.cp_rank_scheduled_tokens[req_id] = self.cp_world_size

        # No sync needed (single DP or test environment without dp_group).
        if self.cp_sync is None:
            self._preempted_this_step.clear()
            return output

        # Participate in the all_reduce even when this rank has no active CP
        # requests so peer ranks are not blocked waiting for this participant.
        if not self.active_cp_requests:
            self.cp_sync.sync_empty()
            self._preempted_this_step.clear()
            return output

        # Build per-request status vector for the all_reduce.
        active_ids = sorted(self.active_cp_requests.keys())
        status: list[int] = []
        finished_ids: list[str] = []
        for req_id in active_ids:
            if req_id not in self.requests:
                # This rank already finished; report SCHEDULED so the peer is
                # not held back. Clean up after the sync.
                s = SCHEDULED
                s_str = "FINISHED(report SCHEDULED)"
                finished_ids.append(req_id)
            elif req_id in output.num_scheduled_tokens:
                s = SCHEDULED
                s_str = "SCHEDULED"
            elif req_id in self._preempted_this_step:
                s = PREEMPTED
                s_str = "PREEMPTED"
            else:
                s = NOT_SCHEDULED
                s_str = "NOT_SCHEDULED"
            req = self.active_cp_requests[req_id]
            logger.debug(
                "[DYCP Debug] CP sync rank=%d req=%s local_status=%s "
                "num_computed=%d in_requests=%s",
                self.cp_rank, req_id, s_str,
                req.num_computed_tokens, req_id in self.requests,
            )
            status.append(s)

        confirmed, soft_rollback_ids, hard_rollback_ids = (
            self.cp_sync.sync_schedule_confirm(active_ids, status)
        )

        # Clean up requests that finished on this rank before the sync.
        for req_id in finished_ids:
            req = self.active_cp_requests.pop(req_id, None)
            self.prev_step_scheduled_req_ids.discard(req_id)
            if req is not None:
                logger.info(
                    "[DYCP Debug] CP request %s removed from active_cp_requests"
                    " on rank %d (finished, peer notified via SCHEDULED)",
                    req_id, self.cp_rank,
                )

        if soft_rollback_ids:
            output = self._soft_rollback(output, soft_rollback_ids)
        if hard_rollback_ids:
            output = self._hard_rollback(output, hard_rollback_ids)

        self._preempted_this_step.clear()

        output.cp_req_ids_sorted = sorted(confirmed) if confirmed else None

        # Signal workers to run a dummy forward for collective ops when this
        # rank has no tokens but peer ranks have confirmed CP tokens.
        if output.total_num_scheduled_tokens == 0 and confirmed:
            output.none_tokens_in_peer_sched = True

        return output

    def _soft_rollback(
        self, output: SchedulerOutput, rollback_ids: list[str]
    ) -> SchedulerOutput:
        """Remove from output and requeue for next step."""
        for req_id in rollback_ids:
            # If the request has already finished on this rank (peer completed
            # one step earlier and update_from_output cleaned self.requests),
            # skip the rollback mechanics and just drop it from active tracking.
            if req_id not in self.requests:
                self.active_cp_requests.pop(req_id, None)
                self.prev_step_scheduled_req_ids.discard(req_id)
                continue
            logger.debug(f"[DYCP Debug] Soft rollback triggered for req {req_id}.")
            if req_id in output.num_scheduled_tokens:
                logger.debug(
                    "[DYCP Debug] Soft rollback req=%s (was SCHEDULED on this rank,"
                    " num_computed=%d)",
                    req_id,
                    self.active_cp_requests[req_id].num_computed_tokens,
                )
                num_tokens = output.num_scheduled_tokens.pop(req_id)
                output.total_num_scheduled_tokens -= num_tokens
                self._remove_req_from_output(output, req_id)

                request = self.active_cp_requests[req_id]
                self.kv_cache_manager.free(request)

                request.status = RequestStatus.WAITING
                # Must reset to 0 even though the intent of soft rollback is to
                # preserve prefill progress. kv_cache_manager.free() releases
                # all blocks for this request. If num_computed_tokens were left
                # at its historical value (e.g. 50K), the base scheduler would
                # take the `else` branch (the KVTransfer path) on the next step
                # and use that stale value directly, yielding num_new_tokens=0
                # and hitting `assert num_new_tokens > 0`. Resetting to 0 forces
                # the base scheduler to call get_computed_blocks() instead, which
                # re-hits the prefix cache and recovers the same progress without
                # redundant computation.
                request.num_computed_tokens = 0
                if request in self.running:
                    self.running.remove(request)
                self.waiting.prepend_request(request)
            else:
                logger.debug(
                    "[DYCP Debug] Soft rollback req=%s (was NOT_SCHEDULED on this"
                    " rank, num_computed=%d, in_running=%s, in_waiting=%s)",
                    req_id,
                    self.active_cp_requests[req_id].num_computed_tokens,
                    self.active_cp_requests[req_id] in self.running,
                    self.active_cp_requests[req_id] in self.waiting,
                )

            # Remove from prev_step_scheduled_req_ids so that next step
            # treats this request as freshly resumed rather than continuing
            # from a previous scheduled step. Without this, the base
            # scheduler's assert not scheduled_in_prev_step fires when the
            # request surfaces from waiting into scheduled_resumed_reqs.
            self.prev_step_scheduled_req_ids.discard(req_id)

        return output

    def _hard_rollback(
        self, output: SchedulerOutput, rollback_ids: list[str]
    ) -> SchedulerOutput:
        """Full rollback: all ranks preempt, reset num_computed_tokens=0."""
        for req_id in rollback_ids:
            # Same as _soft_rollback: skip and clean up if already finished.
            if req_id not in self.requests:
                self.active_cp_requests.pop(req_id, None)
                self.prev_step_scheduled_req_ids.discard(req_id)
                continue
            logger.debug(f"[DYCP Debug] Hard rollback triggered for req {req_id}.")
            if req_id in output.num_scheduled_tokens:
                num_tokens = output.num_scheduled_tokens.pop(req_id)
                output.total_num_scheduled_tokens -= num_tokens
                self._remove_req_from_output(output, req_id)

            request = self.active_cp_requests[req_id]

            if req_id not in self._preempted_this_step:
                # schedule() did not preempt this rank; do it manually.
                self.kv_cache_manager.free(request)
                if request in self.running:
                    self.running.remove(request)

            request.num_computed_tokens = 0
            # Use WAITING (not PREEMPTED) regardless of whether schedule()
            # already preempted this rank. CP rollbacks always happen before
            # execute_model, so the worker has never seen this request's KV
            # state. PREEMPTED would tell the base scheduler to send this as
            # a resumed request, causing a KeyError in the model runner.
            request.status = RequestStatus.WAITING

            # Keep in active_cp_requests; re-queue for the next step.
            if request not in self.waiting:
                self.waiting.prepend_request(request)

            # Remove from prev_step_scheduled_req_ids so that next step
            # does not treat this request as continuing from a previous
            # scheduled step, which would trigger assert failures in the
            # base scheduler's _make_cached_request_data.
            self.prev_step_scheduled_req_ids.discard(req_id)

        return output

    def _remove_req_from_output(
        self, output: SchedulerOutput, req_id: str
    ) -> None:
        """Remove a request from all SchedulerOutput fields."""
        output.scheduled_new_reqs = [
            r for r in output.scheduled_new_reqs if r.req_id != req_id
        ]

        cached = output.scheduled_cached_reqs
        if req_id in cached.req_ids:
            idx = cached.req_ids.index(req_id)
            cached.req_ids.pop(idx)
            if cached.new_token_ids:
                cached.new_token_ids.pop(idx)
            cached.new_block_ids.pop(idx)
            cached.num_computed_tokens.pop(idx)
            cached.num_output_tokens.pop(idx)
            cached.resumed_req_ids.discard(req_id)
            if req_id in cached.all_token_ids:
                del cached.all_token_ids[req_id]

        if output.cp_rank_scheduled_tokens and req_id in output.cp_rank_scheduled_tokens:
            del output.cp_rank_scheduled_tokens[req_id]
        if output.cp_rank_to_req_id and req_id in output.cp_rank_to_req_id:
            output.cp_rank_to_req_id.remove(req_id)
        if output.req_id_to_cp_size and req_id in output.req_id_to_cp_size:
            del output.req_id_to_cp_size[req_id]
        output.num_cp_request = max(0, output.num_cp_request - 1)

        if output.scheduled_spec_decode_tokens and req_id in output.scheduled_spec_decode_tokens:
            del output.scheduled_spec_decode_tokens[req_id]

    # ------------------------------------------------------------------
    # Update from output: handle CP request completion
    # ------------------------------------------------------------------

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Update scheduler state from model output."""
        result = super().update_from_output(scheduler_output, model_runner_output)

        # Do NOT clean up active_cp_requests here. Removal is handled in
        # schedule() once we detect req_id not in self.requests.
        # Removing here would make active_ids diverge between ranks on the
        # next step (one rank finishes a step earlier), causing slot mismatches
        # in the all-reduce and breaking the sync protocol.

        return result
