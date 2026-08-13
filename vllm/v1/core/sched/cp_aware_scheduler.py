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

from typing import TYPE_CHECKING, Any

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
_MAX_CP_SYNC_SLOTS = 32 # TODO [DyCP] 不能动态设置槽位吗？固定数量槽位的，共识时只有前_MAX_CP_SYNC_SLOTS个长请求可以被共识，如果某个rank上超过这个槽位上限位置的请求被调度到，这个请求就不会在所有rank上进行共识，会出错吗？

# Three-state encoding for post-schedule consensus.
SCHEDULED = 2
NOT_SCHEDULED = 1
PREEMPTED = 0


class CPSyncProtocol:
    """Post-schedule consensus protocol for CP request scheduling.

    Uses the DyCP subgroup (``dycp_group``) all-reduce MIN to agree on whether
    each active CP request was successfully scheduled on ALL ranks of its CP
    group in the current step.

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
        dycp_group: "ProcessGroup",
        cp_world_size: int,
        cp_rank: int,
    ):
        # The consensus collective runs inside a single DyCP subgroup
        # (dycp_size ranks), NOT the full DP group: only the ranks that
        # cooperatively decode a given CP request need to agree on its
        # schedule/rollback decision. Using the full DP group here was a
        # leftover bug that forced CP requests to be broadcast to every DP
        # rank (to keep all_reduce slots aligned) and produced duplicate
        # outputs.
        self.dycp_group = dycp_group
        self.cp_world_size = cp_world_size   # TODO [DyCP] self.cp_world_size 参数未用到，删除？
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
        num_slots = min(len(active_ids), _MAX_CP_SYNC_SLOTS) # TODO [DyCP] 不同rank上，active_ids是一致的吗？不一致是否会有影响
        if num_slots == 0:
            return [], [], []

        self._confirm_tensor.zero_()
        for i in range(num_slots):
            self._confirm_tensor[i] = status[i]

        torch.distributed.all_reduce(
            self._confirm_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.dycp_group,
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
            logger.info(
                "[DYCP] CP confirm: confirmed=%d soft_rollback=%d hard_rollback=%d"
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
            group=self.dycp_group,
        )


class CPAwareScheduler(Scheduler):
    """Scheduler with CP awareness for distributed DYCP.

    Each DP rank runs its own CPAwareScheduler. CP requests are coordinated
    via a distributed sync protocol scoped to a single DyCP subgroup
    (``dycp_group``) -- the ``dycp_size`` ranks that cooperatively decode a
    given CP request -- NOT the full DP group.

    Key differences from base Scheduler:
    - Classifies requests as long (CP) or short (DP) based on token threshold
    - CP requests are activated immediately on arrival (the client routes a
      CP request to the dycp_group that owns it, so no announce phase needed)
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
        hash_block_size: int | None = None,
        dycp_group: "ProcessGroup | None" = None,
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

        logger.info("[DYCP] Initializing CPAwareScheduler.....")

        assert dycp_group is not None, (
            "DyCP subgroup must not be none in CPAwareScheduler"
        )

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

        # CP sync protocol (None if dycp_group not provided, e.g., single DP).
        self.cp_sync: CPSyncProtocol | None = None
        if dycp_group is not None and self.cp_world_size > 1:
            self.cp_sync = CPSyncProtocol(
                dycp_group=dycp_group,
                cp_world_size=self.cp_world_size,
                cp_rank=self.cp_rank,
            )
            logger.info(
                "[DYCP] CP consensus scoped to a single DyCP subgroup on "
                "dp_rank=%d: cp_rank=%d, cp_world_size=%d. This rank %s the "
                "CP output stream for CP requests it holds.",
                vllm_config.parallel_config.data_parallel_rank,
                self.cp_rank, self.cp_world_size,
                "owns (emits)" if self.cp_rank == 0 else "suppresses",
            )
        else:
            logger.info(
                "[DYCP] CP consensus disabled on dp_rank=%d "
                "(dycp_group=%s, cp_world_size=%d)",
                vllm_config.parallel_config.data_parallel_rank,
                dycp_group, self.cp_world_size,
            )

        # [DyCP] P(prefill/producer) 端长请求由 dycp 组内 cp_rank=0..N-1 协同 prefill，
        # 各 rank 各持自己 block_id 段（block_id 不跨 rank 协调）。funnel 只让
        # cp_rank=0 发 EngineCoreOutput 会导致其它 cp_rank 的 block_id 丢失，D 只拉
        # 一段 KV 而错答。此处缓冲各 cp_rank 上报的 per-req block_id 段，由
        # owner(cp_rank=0) 在 update_from_output 经 gloo all_gather 收齐后拼成
        # per-cp_rank 多段 remote_block_ids 一份传 D。
        # 结构: {req_id: {cp_rank: remote_block_ids_segment}}（segment 为 BlockIds，
        # per-group）。req 全 cp_rank 段到齐后由 owner 拼多段并清理。
        self._dycp_block_shards: dict[str, dict[int, Any]] = {}
        self._gather_enabled = (
            self.cp_sync is not None
            and self.cp_world_size > 1
            and vllm_config.kv_transfer_config is not None
            and vllm_config.kv_transfer_config.is_kv_producer
        )

    # ------------------------------------------------------------------
    # Request classification and routing
    # ------------------------------------------------------------------

    def _is_long_request(self, request: Request) -> bool:
        """Classify request as long (CP) based on prefill token count."""
        num_prefill_tokens = request.num_tokens - request.num_output_tokens

        logger.info(f"[DYCP] Test: request.num_tokens={request.num_tokens}, request.num_output_tokens={request.num_output_tokens}, num_prefill_tokens={num_prefill_tokens}, self.long_request_threshold={self.long_request_threshold} .....")

        return num_prefill_tokens >= self.long_request_threshold
    
    def _get_local_cp_tokens(self, total_tokens: int) -> int:
        """Calculate how many tokens this rank stores for a CP request."""
        base = total_tokens // self.cp_world_size
        remainder = total_tokens % self.cp_world_size
        return base + (1 if self.cp_rank < remainder else 0)

    def _cp_width_of(self, request: Request) -> int:
        """CP shard width: long request = cp_world_size (dycp_size), short = 1.

        Based on len(request.cp_ranks) (set in add_request: long =
        list(range(cp_world_size)), short = [cp_rank]). Used for per-request KV
        block accounting on the scheduler side -- long requests are accounted
        by this rank's local shard, short requests by the full num_tokens
        (long/short split). Always 1 when DyCP is off (cp_world_size <= 1).
        """
        if self.cp_world_size <= 1:
            return 1
        cp_ranks = getattr(request, "cp_ranks", None)
        return len(cp_ranks) if cp_ranks else 1
    
    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Add request, routing CP requests directly to active state."""
        if self.cp_world_size <= 1 or not self._is_long_request(request):
            logger.info(f"[DYCP] Test: self.cp_world_size <= 1: {self.cp_world_size <= 1}.....")
            logger.info(f"[DYCP] Test: self._is_long_request(request): {self._is_long_request(request)}.....")


            request.cp_ranks = [self.cp_rank]

            logger.info(f"[DYCP] Test: Add short request 添加短请求: {request.request_id}.....")

            super().add_request(request)
        else:
            # Client guarantees CP requests are broadcast to all ranks, so
            # activate immediately without a pending/announce phase.
            logger.info("[DYCP] Test: Long request detected,添加长请求 request_id = %s.", request.request_id)
            request.cp_ranks = list(range(self.cp_world_size))
            self.active_cp_requests[request.request_id] = request
            self.requests[request.request_id] = request
            self.waiting.add_request(request)
            logger.info(
                "[DYCP] CP request %s activated on cp rank %d (tokens=%d)",
                request.request_id,
                self.cp_rank,
                request.num_tokens,
            )

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Override to track CP requests preempted by schedule()."""
        logger.info("[DYCP] Preempt request, request_id = %s.", request.request_id)
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
        # [DyCP] B2 回退说明: 不再以 _build_kv_connector_meta 覆写提前填 CP 字段
        # (修法A 实测会激活 mooncake 的 reqs_in_batch 过滤③, 致生产者侧把长 CP 请求
        # 塞入 meta.reqs_in_batch 却不进 requests_to_send, 滞留不清除、锁死 soft-rollback
        # idle 死循环, 最终 sample_tokens 超时、服务挂, 见 v61)。已移除覆写(B2), 
        # 这些字段恢复为仅在此处(schedule 末尾)赋值, build_connector_meta 读到的恒为
        # dataclass 默认值(cp_rank=0/num_cp_request=0/cp_req_id=None->[]), 行为回到
        # 修法b 之后的稳态(即 v60 的 8/8 全对)。字段名 cp_rank_to_req_id 已更名为
        # cp_req_id, 仅改名为惰性、无行为影响。TODO 见 output.py cp_req_id 字段注释。
        output.cp_rank = self.cp_rank
        cp_req_ids: list[str] = []
        req_id_to_cp_size: dict[str, list[int]] = {}
        for req_id in output.num_scheduled_tokens:
            if req_id in self.active_cp_requests:
                logger.info(f"[DYCP] Test: scheduler 进入长序列处理逻辑.....")
                cp_req_ids.append(req_id)
                req_id_to_cp_size[req_id] = self.active_cp_requests[req_id].cp_ranks
        output.num_cp_request = len(cp_req_ids)
        output.cp_req_id = cp_req_ids if cp_req_ids else None
        output.req_id_to_cp_size = req_id_to_cp_size if req_id_to_cp_size else None
        if output.cp_rank_scheduled_tokens is None:
            output.cp_rank_scheduled_tokens = {}

        cp_req_set = set(cp_req_ids)
        for req_id in output.num_scheduled_tokens:
            output.cp_rank_scheduled_tokens[req_id] = (self.cp_world_size if req_id in cp_req_set else 1) # TODO [DyCP] 从变量定义来看，应该是记录该CP rank上对每个长请求调度的token数，为什么赋值是cp_world_size？还是变量命名有误？
            # output.cp_rank_scheduled_tokens[req_id] = self.cp_world_size

        # [DYCP] Step1 diag: long/short split counts this step (verify
        # _cp_width_of distinguishes long=cp_world_size vs short=1).
        num_dp_req = len(output.num_scheduled_tokens) - output.num_cp_request
        logger.info(
            "[DYCP] step cp_width summary: rank=%d cp_world_size=%d "
            "num_cp_req=%d num_dp_req=%d",
            self.cp_rank, self.cp_world_size, output.num_cp_request, num_dp_req,
        )

        # No sync needed (single DP or test environment without dp_group).
        if self.cp_sync is None:
            self._preempted_this_step.clear()
            return output

        if not self.active_cp_requests:
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
            logger.info(
                "[DYCP] CP sync rank=%d req=%s local_status=%s "
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
                    "[DYCP] CP request %s removed from active_cp_requests"
                    " on rank %d (finished, peer notified via SCHEDULED)",
                    req_id, self.cp_rank,
                )

        if soft_rollback_ids:
            output = self._soft_rollback(output, soft_rollback_ids)
        if hard_rollback_ids:
            output = self._hard_rollback(output, hard_rollback_ids)

        self._preempted_this_step.clear()

        # output.cp_req_ids_sorted = sorted(confirmed) if confirmed else None

        # Signal workers to run a dummy forward for collective ops when this
        # rank has no tokens but peer ranks have confirmed CP tokens.
        if output.total_num_scheduled_tokens == 0 and confirmed:
            output.none_tokens_in_peer_sched = True   # TODO [DyCP] 这里为什么？？？？


        logger.info(f"[DYCP] Test: 44444444: CPAwareScheduler调度结束.....")

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
            logger.debug(f"[DYCP] Soft rollback triggered for req {req_id}.")
            if req_id in output.num_scheduled_tokens:
                # [DyCP] 设计语义: 长请求在子组某个 dp 本拍未调度到时, 本拍调度到
                # 它的 dp 仅把该请求从本次 scheduler_output 剔除, 已计算的 KV 与
                # prefill 进度(num_computed_tokens)保持不变。故本分支只做"剔除本次
                # 调度结果", 不释放 KV、不重置进度、不降级状态:
                #   - pop num_scheduled_tokens 并扣减 total_num_scheduled_tokens;
                #     _remove_req_from_output 同步清理 scheduled_new_reqs /
                #     scheduled_cached_reqs(new_block_ids 等) / cp_rank_scheduled_tokens
                #     / cp_req_id / req_id_to_cp_size / num_cp_request /
                #     scheduled_spec_decode_tokens 等全部 output 侧计数。
                #   - 不调用 kv_cache_manager.free(): 那会释放该请求全部历史块,
                #     违背"已计算的 KV 不动", 且会破坏正在向 D 传输的 KV。
                #   - 不重置 num_computed_tokens、不把 status 降级为 WAITING、不移出
                #     running: prefill 进度与状态原样保留, 下一拍续算即可。
                # 本拍为该请求新分配的尾部块(尚未写入, soft_rollback 发生在
                # execute_model 之前)予以保留: 它们不属"已计算 KV", 下一拍真正执行
                # 时写入并复用。FullAttentionManager 对 running 请求的
                # get_num_blocks_to_allocate / allocate_new_blocks 在块已足够时均
                # max(...,0) 钳到 0, 故保留尾部块不触发负分配或断言; attention 的
                # causal mask 只读 num_computed..num_computed+num_new, 不脏读。
                num_tokens = output.num_scheduled_tokens.pop(req_id)
                output.total_num_scheduled_tokens -= num_tokens
                self._remove_req_from_output(output, req_id)
                request = self.active_cp_requests[req_id]
                logger.info(
                    "[DYCP] Soft rollback req=%s (was SCHEDULED on this rank, "
                    "num_computed=%d) -> 仅剔除本次 output, 保留已算 KV 与进度, "
                    "status 保持 RUNNING",
                    req_id,
                    request.num_computed_tokens,
                )
            else:
                logger.info(
                    "[DYCP] Soft rollback req=%s (was NOT_SCHEDULED on this"
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
            logger.info(f"[DYCP] Hard rollback triggered for req {req_id}.")
            if req_id in output.num_scheduled_tokens:
                num_tokens = output.num_scheduled_tokens.pop(req_id)
                output.total_num_scheduled_tokens -= num_tokens
                self._remove_req_from_output(output, req_id)

            request = self.active_cp_requests[req_id]

            if req_id not in self._preempted_this_step:
                # schedule() did not preempt on this rank; do it manually.
                logger.info(f"[DYCP] Schedule() did not preempt on this rank; do it manually for req {req_id}.")
                self.kv_cache_manager.free(request)
                if request in self.running:
                    self.running.remove(request)

            request.num_computed_tokens = 0
            # Use WAITING (not PREEMPTED) regardless of whether schedule()
            # already preempted this rank. CP rollbacks always happen before
            # execute_model, so the worker has never seen this request's KV
            # state. PREEMPTED would tell the base scheduler to send this as
            # a resumed request, causing a KeyError in the model runner.
            request.status = RequestStatus.WAITING   # TODO [DyCP] 请求被抢占了为什么是设置回waitting状态

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
        if output.cp_req_id and req_id in output.cp_req_id:
            output.cp_req_id.remove(req_id)
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

        # [DyCP] 阶段1: P 端 owner 聚合各 cp_rank 的 block_id 段。
        # 各 cp_rank 各自 prefill 长 prompt 的一段、各持自己 block_id（block_id 不跨
        # rank 协调）。本步若有 CP req finish，本 rank 把它在该 req 上的
        # remote_block_ids 段经 gloo all_gather 交换给全组；owner(cp_rank=0) 收齐后
        # 拼成 per-cp_rank 多段写进 owner 的 kv_transfer_params，使 D 能多源拉全 KV。
        # 对称性：仅当本组仍有 active CP 请求（或本步刚 finish CP req）时才进 gather，
        # 该条件由 CP sync 保证两组同 true/false，避免 all_gather 错步死锁。
        self._maybe_gather_dycp_block_shards(result)

        # Do NOT clean up active_cp_requests here. Removal is handled in
        # schedule() once we detect req_id not in self.requests.
        # Removing here would make active_ids diverge between ranks on the
        # next step (one rank finishes a step earlier), causing slot mismatches
        # in the all-reduce and breaking the sync protocol.

        # [DyCP] A CP (long) request is decoded cooperatively by the `dycp_size`
        # ranks of its CP group. The worker aligns the sampled tokens inside
        # the group (PCPManager all_gather + `_sync_dycp_sampled_token_ids`
        # broadcast src=0), but every rank in the group still independently
        # emits an EngineCoreOutput for the request, so without suppression the
        # client sees a duplicate token stream per CP request. Funnel the CP
        # request's token/finish output to a single owner -- the cp_rank 0 of
        # the group (== the broadcast src=0, whose tokens are authoritative).
        # Non-owner ranks still run the full compute / consensus / KV-put here;
        # only the *output* is muted.
        if self.active_cp_requests and self.cp_rank != 0:
            cp_req_ids = set(self.active_cp_requests)
            suppressed = False
            for eco in result.values():
                if eco.outputs:
                    before = len(eco.outputs)
                    eco.outputs = [
                        o for o in eco.outputs
                        if o.request_id not in cp_req_ids
                    ]
                    if len(eco.outputs) < before:
                        suppressed = True
                if eco.finished_requests:
                    non_cp_finished = eco.finished_requests - cp_req_ids
                    eco.finished_requests = (
                        non_cp_finished if non_cp_finished else None
                    )
            if suppressed:
                # Non-owner CP rank: muted the CP request's token/finish output;
                # cp_rank 0 of this subgroup is the sole emitter.
                logger.info(
                    "[DYCP] cp_rank=%d (non-owner) suppressed CP outputs for "
                    "req_ids=%s this step.",
                    self.cp_rank, sorted(cp_req_ids),
                )

        # [DyCP] 发送端统一: 把所有 D-bound 输出的 remote_block_ids 规整为
        # per-cp_rank canonical 多段, 使 D 端无需结构判别(见下方方法)。
        self._normalize_dycp_remote_block_ids(result)

        return result

    def _normalize_dycp_remote_block_ids(
        self, result: dict[int, EngineCoreOutputs]
    ) -> None:
        """[DyCP] 发送端(P)统一 remote_block_ids 结构为 per-cp_rank canonical 多段。

        根因: P 端对 remote_block_ids 历史上有两种产出形态, 是 v48/v50 D 端反复
        打补丁的根结:
          (1) per-cp_rank 多段(3 层): 经 _maybe_gather_dycp_block_shards 合并后,
              owner(cp_rank=0) 写回 [seg0, seg1, ...], len == len(remote_dycp_ranks)。
          (2) per-group 扁平(2 层): 未合并时(短请求不经 gather; 或长请求因部分 cp_rank
              hard rollback 致 gather 收不齐段), 由 _connector_finished 直接产出 ([1],)。
        两态混发使 D 端 _get_kv_split_metadata 须猜结构: 按 len(remote_dycp_ranks)>1
        判别会误判回滚单 owner 的多 rank 请求(v50); 按 owner 槽位扩段又会用 rank 值
        当索引、误清空短请求 block_ids(v51 S0, owner_slot=1 != 槽 0)。

        统一方案: 在发送的唯一公共出口(update_from_output, funnel 之后)把所有
        D-bound 输出的 flat(2 层) 一律规整为 3 层 canonical:
            [(flat) if i == 0 else empty_seg for i in range(len(remote_dycp_ranks))]
        依据不变式: 发射该 flat 的 rank 恒在 remote_dycp_ranks 的 index 0——
          短请求: remote_dycp_ranks = [sole_cp_rank], sole 在 index 0;
          长请求: remote_dycp_ranks = list(range(cp_world_size)) = [0,1,...],
                  发射者 owner = cp_rank=0 在 index 0(hard rollback 时也仅 owner 发射)。
        已是 3 层(gather 合并)则不动。规整后 D 端恒以
        meta_remote_block_ids[cp_rank_idx][group_idx] 索引, 无需任何结构判别。
        """
        for eco in result.values():
            for o in eco.outputs:
                p = o.kv_transfer_params
                if not p:
                    continue
                rb = p.get("remote_block_ids")
                if rb is None:
                    continue
                ranks = p.get("remote_dycp_ranks") or [0]
                n = len(ranks)
                if n == 0:
                    continue
                # 判别 3 层(per-cp_rank) vs 2 层(per-group flat):
                # 段首非空 group 元素是 list -> 3 层; 是 int(block_id) -> 2 层 flat。
                is_3level = False
                try:
                    for seg in rb:
                        if not hasattr(seg, "__len__") or len(seg) == 0:
                            continue
                        is_3level = hasattr(seg[0], "__len__")
                        break
                except (TypeError, IndexError):
                    is_3level = False
                if is_3level:
                    continue
                # flat(2 层) -> canonical 3 层, 发射段恒在 index 0, 其余槽位置空段。
                num_groups = len(rb)
                empty_seg = tuple([] for _ in range(num_groups))
                canon = [rb if i == 0 else empty_seg for i in range(n)]
                p["remote_block_ids"] = canon

    def _maybe_gather_dycp_block_shards(
        self, result: dict[int, EngineCoreOutputs]
    ) -> None:
        """[DyCP] P 端: 各 cp_rank 经 gloo all_gather 交换本步 finish 的 CP req
        block_id 段, owner(cp_rank=0) 收齐拼 per-cp_rank 多段写回 kv_transfer_params。

        见 __init__ 的 _dycp_block_shards 注释。对称性: 仅当本组仍有 CP 请求在跑时
        才进 gather, 该条件由 CP sync 保证两组同 true/false, 避免错步死锁。

        数据变化示例 (dycp_size=4, req-A prefill 完成, 各 cp_rank 持各自独立
        block pool 分配的 block_id 段, block_id 不跨 rank 协调):

          初始 — 4 个独立进程各持自己的段:
            cp_rank=0: 段0=([1],)   cp_rank=1: 段1=([2],)
            cp_rank=2: 段2=([3],)   cp_rank=3: 段3=([4],)

          步骤1 — 各 rank 收集本进程的 my_finished (只含自己那段):
            rank0: {"req-A": ([1],)}   rank1: {"req-A": ([2],)}
            rank2: {"req-A": ([3],)}   rank3: {"req-A": ([4],)}

          步骤2 — gloo all_gather_object 交换后, 4 个进程都拿到全量:
            gathered = [
              {"req-A": ([1],)},   # index 0 = cp_rank=0
              {"req-A": ([2],)},   # index 1 = cp_rank=1
              {"req-A": ([3],)},   # index 2 = cp_rank=2
              {"req-A": ([4],)},   # index 3 = cp_rank=3
            ]

          步骤3 — 合并进跨步缓冲 _dycp_block_shards:
            _dycp_block_shards = {
              "req-A": {0: ([1],), 1: ([2],), 2: ([3],), 3: ([4],)}
            }
            len(shards)==4==cp_world_size → 拼齐, newly_complete=["req-A"]

          步骤4 — 所有 rank 一致 pop (缓冲同步变空, 防下一步不对称死锁);
            owner(cp_rank=0) 额外写回自己的 EngineCoreOutput:
              remote_block_ids = [([1],), ([2],), ([3],), ([4],)]
              ↑ per-cp_rank 多段 list, 非 owner 不写回 (output 被 funnel suppress)

          D 端 — funnel 只发 cp_rank=0 的 output, D 收到多段:
            remote_block_ids = [([1],), ([2],), ([3],), ([4],)]
            remote_dycp_ranks = [0, 1, 2, 3]
            按 remote_dycp_ranks 拆 4 shard, 各从对应 P rank 的 host/port
            mooncake RDMA 拉 block 1/2/3/4 的 KV, 4 源拼全 KV → D decode 正确。

        跨步缓冲 (防御性): 正常 CP sync 保证 4 rank 同步 finish, 同一步全段到齐。
        若极端某 rank 段延迟一步, 暂存缓冲, 下步继续 gather 直到拼齐——但对称性
        要求同组 rank 同时进/退 gather, 该前提由 CP sync 共识保证。
        """
        if not self._gather_enabled:
            return

        # 1) 收集本 rank 这步 finish 的 CP req -> remote_block_ids 段。
        #    段来自 base scheduler _connector_finished -> connector.request_finished
        #    填进 EngineCoreOutput.kv_transfer_params['remote_block_ids']。
        my_finished: dict[str, Any] = {}
        cp_ids = set(self.active_cp_requests)
        for eco in result.values():
            for o in eco.outputs:
                if (o.request_id in cp_ids
                        and o.kv_transfer_params is not None
                        and o.kv_transfer_params.get("remote_block_ids") is not None):
                    my_finished[o.request_id] = o.kv_transfer_params["remote_block_ids"]

        # 对称条件: 仅当本步有新 finish 的 CP req, 或缓冲中仍有未拼齐段时才 gather。
        # 注意: active_cp_requests 在 req finish 后可能仍非空(P 端 D 未 ack), 故不能
        # 用它作每步 gather 触发, 否则 finish 后每步空 gather 会因两 rank 调用节奏
        # 不一致而 gloo 对称性死锁。两 rank 对 my_finished 非空/缓冲非空的判定在
        # 同步 finish 下一致(见 v14 证实两 cp_rank 同步 finish), 错步时另作兜底。
        if not my_finished and not self._dycp_block_shards:
            return

        # 2) gloo all_gather 交换各 rank 这步的 finish 字典(对称调用)。
        gathered: list[dict[str, Any]] = [{} for _ in range(self.cp_world_size)]
        torch.distributed.all_gather_object(
            gathered, my_finished, group=self.cp_sync.dycp_group,
        )

        # 3) 合并进 per-req 缓冲 {req_id: {cp_rank: seg}}。
        newly_complete: list[str] = []
        for rank, fin in enumerate(gathered):
            for req_id, seg in fin.items():
                shards = self._dycp_block_shards.setdefault(req_id, {})
                if rank not in shards:
                    shards[rank] = seg
        for req_id, shards in self._dycp_block_shards.items():
            if len(shards) == self.cp_world_size:
                newly_complete.append(req_id)

        # 4) 已拼齐的 req 清缓冲: 所有 rank 一致清理(不只 owner), 保证两 rank
        #    _dycp_block_shards 同步变空, 避免 finish 后 non-owner 缓冲残留导致
        #    gather 触发不对称而死锁。owner 额外把多段写回 kv_transfer_params。
        for req_id in newly_complete:
            shards = self._dycp_block_shards.pop(req_id)
            if self.cp_rank == 0:
                merged = [shards[r] for r in range(self.cp_world_size)]
                for eco in result.values():
                    for o in eco.outputs:
                        if o.request_id == req_id and o.kv_transfer_params is not None:
                            o.kv_transfer_params["remote_block_ids"] = merged