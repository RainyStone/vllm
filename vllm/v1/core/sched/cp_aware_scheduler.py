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
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
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
        # [DyCP] 共识张量布局(33 元):
        #   索引 [0]           : has_cp 固定标志槽。恒存在、不随 num_slots 截断。
        #                        填 1 表示本拍本 rank 调度到了 CP 长请求, 填 0 表示没有。
        #                        all_reduce MIN 后, 该槽的 min = subgroup_all_schedule_cp_request
        #                        (=1 当且仅当子组所有端本拍都排了 CP 长请求; =0 表示至少一端没排)。
        #                        用于驱动阶段2降级: 本端有长但子组不全有长 -> 降级对齐两端 num_cp。
        #   索引 [1 .. _MAX_CP_SYNC_SLOTS]: CP 请求状态槽(SCHEDULED/NOT_SCHEDULED/PREEMPTED), 语义不变。
        # has_cp 固定放第 0 槽、不截断, 保证空拍(无 active_cp)两端仍同形状(all_reduce 全 33 元),
        # 根治(共识仅忙拍调、空拍不调)导致的 32元↔1元 错配崩溃(v81)与忙端 hang。
        self._confirm_tensor = torch.zeros(
            _MAX_CP_SYNC_SLOTS + 1, dtype=torch.int32, device="cpu"
        )

    def sync_schedule_confirm(
        self,
        active_ids: list[str],
        status: list[int],
        local_has_cp: int,
    ) -> tuple[list[str], list[str], list[str], int]:
        """Post-schedule consensus using three-state encoding, 合并阶段2协商.

        把"CP 请求状态共识"(原 32 槽)与"阶段2 是否含长协商"(原 align_execute_cp
        的 1 元 all_reduce)合并成本方法内的**一次** all_reduce, 根治两者抢同一
        dycp_group、形状 32≠1 错配崩溃(v81)及"共识仅忙拍调、空拍不调"的忙端 hang.

        Args:
            active_ids: Active CP request IDs (sorted, identical across ranks).
            status: Per-request status on this rank
                    (SCHEDULED=2 / NOT_SCHEDULED=1 / PREEMPTED=0)。
            local_has_cp: 本拍本 rank 是否调度到 CP 长请求(1=是, 0=否)。
                          写入固定槽 [0], all_reduce MIN 后该槽即阶段2协商结果。

        Returns:
            (confirmed_ids, soft_rollback_ids, hard_rollback_ids,
             subgroup_all_schedule_cp_request)
            其中 subgroup_all_schedule_cp_request = 子组所有端 local_has_cp 的 MIN:
              =1 当且仅当子组所有端本拍都排了 CP 长请求(两端都含长);
              =0 表示至少一端本拍没排 CP 长请求(含长端须降级对齐)。
        """
        num_slots = min(len(active_ids), _MAX_CP_SYNC_SLOTS) # TODO [DyCP] 不同rank上，active_ids是一致的吗？不一致是否会有影响

        # [DyCP] 根因修复(v77/v78): CP 槽([1..32])初始填充 NOT_SCHEDULED(1) 而非
        # PREEMPTED(0)。短端 active_ids 较少时尾部超出槽位保持初始值, 若为 0 则
        # all_reduce MIN 会把长端同位置真 req 的 min 拉到 0 -> 误判 hard_rollback ->
        # 释放正在用/正在传 D 的 KV 块 -> Worker KeyError(v77)/ 卡死(v78)。
        # has_cp 固定槽 [0] 单独写 local_has_cp(不随 num_slots 截断)。
        self._confirm_tensor.fill_(NOT_SCHEDULED)
        self._confirm_tensor[0] = local_has_cp
        for i in range(num_slots):
            self._confirm_tensor[i + 1] = status[i]

        # ★ 关键改: 空拍(num_slots==0)不再提前 return —— 仍走 all_reduce 全 33 元。
        # 否则忙端共识(33元)等不到空端 -> hang; 且原 align 的 1 元与共识 32 元错配 -> 崩。
        torch.distributed.all_reduce(
            self._confirm_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.dycp_group,
        )

        subgroup_all_schedule_cp_request = int(self._confirm_tensor[0].item())

        confirmed: list[str] = []
        soft_rollback: list[str] = []
        hard_rollback: list[str] = []
        if num_slots > 0:
            for i in range(num_slots):
                val = self._confirm_tensor[i + 1].item()
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

        return confirmed, soft_rollback, hard_rollback, subgroup_all_schedule_cp_request
    
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

        # [DyCP v82 保序修复] 首次调度被回滚(soft_rollback/degrade)的 CP 请求: 已进
        # running、但 worker 从未收到它的 new_req 注册(degrade/soft_rollback 把它从
        # 当拍 output 剔除了)。若按现行"留 running 续算"处理, 下拍 base 会以
        # scheduled_cached_reqs 下发, worker self.requests[req_id] 查无 -> KeyError
        # (gpu_model_runner._update_states, v82 根因)。
        # 修复: 这类请求留 running 原位(不回 waiting, 保序)并记入本集合; 下拍它被
        # base 当 cached 续算 schedule 时, 若子组已对齐(未被本拍再次回滚), 由
        # _emit_pending_new_cp_as_new 把它从 scheduled_cached_reqs 挪到
        # scheduled_new_reqs 发出 -> worker 走 new 注册路径, 不再 KeyError, 且
        # running 位置不动 -> 无持续失序。
        self._cp_reqs_pending_new_emit: set[str] = set()

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

        # 本拍本 rank 是否调度到 CP 长请求(1=是, 0=否)。
        # 用于合并共识 sync_schedule_confirm 的 has_cp 固定槽 [0]: all_reduce MIN 后
        # 该槽 = subgroup_all_schedule_cp_request(子组是否所有端本拍都含长),
        # 驱动下方阶段2降级。须在 CP 元数据标注(num_cp_request 已赋)之后计算。
        local_has_cp = 1 if output.num_cp_request > 0 else 0

        # No sync needed (single DP or test environment without dp_group).
        if self.cp_sync is None:
            self._preempted_this_step.clear()
            return output

        # [DyCP] 合并修复(v81 根因): 不再因"本端无 active_cp"就提前 return。
        # 空拍也必须进 sync_schedule_confirm 的 33 元 all_reduce: 否则忙端
        # (有 active_cp)的共识会一直等空端 -> hang; 且原"空拍跳过共识、忙端调
        # 共识(32元)"配合 align_execute_cp 的独立 1 元 all_reduce, 形状 32!=1
        # 错配抢同一 dycp_group -> gloo Connection reset 崩(v81)。现两者合并成
        # schedule() 内一次 33 元 all_reduce, 空拍端 has_cp=0 + CP 槽全
        # NOT_SCHEDULED, 与忙端同形状配对, 根治 hang 与崩。
        # Build per-request status vector for the all_reduce.
        active_ids = sorted(self.active_cp_requests.keys())
        status: list[int] = []
        finished_ids: list[str] = []
        for req_id in active_ids:
            req = self.active_cp_requests[req_id]
            # [DyCP] 根因修复: CP 请求 finish 后必须退出 active_cp_requests,
            # 否则 has_requests() 恒真、引擎空转、最终 NPU 看门狗超时(v62/v60)。
            # 原仅以 req_id not in self.requests 判"本 rank 已完成", 但 P 端
            # mooncake producer 对 CP prefill 请求 finished_sending 不上报,
            # delay_free 永不释放、请求永不移出 self.requests, active_cp 永不排空。
            # 修正: 增加 req.is_finished() 作为"CP 工作已完成"判据, 与 self.requests
            # 的 KV 块释放生命周期解耦。is_finished() 由 super().update_from_output
            # 在各 cp_rank 同一步同步置位(共识已保证同组同伐), 对称安全; 仍以
            # SCHEDULED 上报、sync 后统一 pop, 保留"一 rank 先完成不拖 peer"的兼容
            # 语义。保留 not in self.requests 分支兼容 D 端/请求被实际移除的既有路径。
            if (req_id not in self.requests) or req.is_finished():
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
            logger.info(
                "[DYCP] CP sync rank=%d req=%s local_status=%s "
                "num_computed=%d in_requests=%s",
                self.cp_rank, req_id, s_str,
                req.num_computed_tokens, req_id in self.requests,
            )
            status.append(s)

        confirmed, soft_rollback_ids, hard_rollback_ids, subgroup_all_schedule_cp_request = (
            self.cp_sync.sync_schedule_confirm(active_ids, status, local_has_cp)
        )

        # Clean up requests that finished on this rank before the sync.
        for req_id in finished_ids:
            req = self.active_cp_requests.pop(req_id, None)
            self.prev_step_scheduled_req_ids.discard(req_id)
            # [DyCP v82] 同步清 pending(首调回滚未发出的 CP 请求若被 finish)
            self._cp_reqs_pending_new_emit.discard(req_id)
            if req is not None:
                logger.info(
                    "[DYCP] CP request %s removed from active_cp_requests"
                    " on rank %d (finished, peer notified via SCHEDULED)",
                    req_id, self.cp_rank,
                )

        # [DyCP 阶段2] 协商降级: 本端含长(local_has_cp=1)但子组不全含长
        # (subgroup_all_schedule_cp_request=0) -> 全量剔除本拍 CP 请求并回退
        # num_computed(降级), 对齐子组两端 num_cp 同 0(都不进 dycp all_gather)。
        # 降级插在 soft/hard_rollback 之前且互斥安全: 降级场景下至少一端空拍
        # (其 has_cp=0、CP 槽全 NOT_SCHEDULED), 该端不会投 PREEMPTED, 故
        # hard_rollback_ids 为空; soft_rollback_ids 含本端全部 cp, 但已被降级从
        # output 剔除(num_scheduled_tokens 已 pop), _soft_rollback 的 SCHEDULED
        # 分支判 not in num_scheduled_tokens 走 else 无害空转, 不重复回退。
        if local_has_cp == 1 and subgroup_all_schedule_cp_request == 0:
            self._degrade_cp_from_output(output)
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

        # [DyCP v82 保序修复] 首次调度被回滚的 CP 请求留 running 原位并标 pending,
        # 本拍若又被 base 当 cached 续算 schedule 且子组已对齐(未被本拍再次回滚),
        # 从 scheduled_cached_reqs 挪到 scheduled_new_reqs 发出(worker 注册),
        # 避免 KeyError, 且 running 位置不动 -> 保序。
        self._emit_pending_new_cp_as_new(output)

        return output
    
    def _soft_rollback(
        self, output: SchedulerOutput, rollback_ids: list[str]
    ) -> SchedulerOutput:
        """Remove from output and requeue for next step."""
        for req_id in rollback_ids:
            # [DyCP] v75 根因修复: 长请求在本 rank 已 finish 时, schedule() 上方
            # 的 finished 清理已将其从 active_cp_requests 移除; 但若此时共识因
            # peer 本拍未调度该请求而把它列入 soft_rollback_ids, 则本 rank 既无
            # active_cp 条目、本次 output 也未含该请求(finished 不再调度), 直接
            # 跳过即可。否则下方 SCHEDULED/NOT_SCHEDULED 两分支 self.
            # active_cp_requests[req_id] 会 KeyError(两端长请求 finish 但被
            # soft_rollback, 上方 finished 清理先 pop 致此处空访 -> EngineCore 崩
            # -> worker 死 -> 对端 metadata AR connection closed by peer 级联)。
            if req_id not in self.active_cp_requests:
                self.prev_step_scheduled_req_ids.discard(req_id)
                logger.info(
                    "[DYCP] Soft rollback skip req=%s on rank %d: not in "
                    "active_cp_requests (already finished/removed by cleanup)",
                    req_id, self.cp_rank,
                )
                continue
            # If the request has already finished on this rank (peer completed
            # one step earlier and update_from_output cleaned self.requests),
            # skip the rollback mechanics and just drop it from active tracking.
            if req_id not in self.requests:
                self.active_cp_requests.pop(req_id, None)
                self.prev_step_scheduled_req_ids.discard(req_id)
                continue
            logger.debug(f"[DYCP] Soft rollback triggered for req {req_id}.")
            if req_id in output.num_scheduled_tokens:
                # [DyCP] soft_rollback 的 SCHEDULED 分支与阶段2降级语义完全一致
                # (剔除本次 output + 回退本拍虚推 num_computed + 保留已算 KV 块 /
                # status RUNNING / running 队列), 共用公共子方法
                # _rollback_cp_req_from_output, 消除"两套剔除代码"分叉。完整设计
                # 语义(KV 不动 / 进度回退 / 尾部块保留 等)详见该方法 docstring。
                self._rollback_cp_req_from_output(output, req_id)
                logger.info(
                    "[DYCP] Soft rollback req=%s (was SCHEDULED on this rank) "
                    "-> 剔除本次 output + 回退本拍虚推 num_computed(execute 前未真算), "
                    "保留已算物理 KV 块, status RUNNING",
                    req_id,
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
            # [DyCP v82] hard_rollback 已把 req 放回 waiting(下拍走 waiting->new
            # 正常路径), 不必再 pending。
            self._cp_reqs_pending_new_emit.discard(req_id)
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
    
    def _remove_req_from_cached_only(
        self, output: SchedulerOutput, req_id: str
    ) -> None:
        """仅从 scheduled_cached_reqs 移除一个 req(平行列表 + all_token_ids +
        resumed_req_ids), 不动 num_scheduled_tokens/total/CP 元数据/
        scheduled_new_reqs。供 cached->new 挪动使用(区别于 _remove_req_from_output
        会连 CP 元数据一起清)。"""
        cached = output.scheduled_cached_reqs
        if req_id not in cached.req_ids:
            return
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

    def _remove_req_from_output(
        self, output: SchedulerOutput, req_id: str
    ) -> None:
        """Remove a request from all SchedulerOutput fields."""
        output.scheduled_new_reqs = [
            r for r in output.scheduled_new_reqs if r.req_id != req_id
        ]
        self._remove_req_from_cached_only(output, req_id)

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
    # [DyCP] CP 请求从 output 回滚的公共子方法: soft_rollback 与 阶段2降级 共用
    # ------------------------------------------------------------------

    def _rollback_cp_req_from_output(
        self, output: SchedulerOutput, req_id: str
    ) -> None:
        """从一个 scheduler_output 中剔除指定 CP 请求 + 回退本拍虚推的 num_computed。

        soft_rollback(共识后本端没分到段)与阶段2降级(peer 无长本端须弃长)在
        num_computed_tokens / KV 块 / status / running 队列的处理上语义完全一致,
        共用本方法, 消除"一个回退一个不回退"的旧分叉。各点语义(与 commit2 一致):
          - pop num_scheduled_tokens + 扣减 total_num_scheduled_tokens;
            _remove_req_from_output 同步清理 scheduled_new_reqs /
            scheduled_cached_reqs / cp_rank_scheduled_tokens / cp_req_id /
            req_id_to_cp_size / num_cp_request / scheduled_spec_decode_tokens 等
            output 侧计数。
          - 回退本拍 _update_after_schedule 乐观推进的 num_computed_tokens
            (max(0, -num_scheduled)): soft_rollback/降级都发生在 execute_model 之前,
            本拍 CP 段实际未执行, num_computed 的本拍推进是调度乐观值而非真实计算,
            不回退会致 base 下拍 num_new_tokens = 0 永远 SKIP -> prefill 永不重排 ->
            永不 execute -> sample 永不做(max_tokens=1 的 sample 在 prefill execute
            拍采) -> active_cp 排不空 -> idle 死循环卡死(v76/v77)。回退只减本拍
            num_scheduled, 保留之前拍真实计算的进度。
          - 不调用 kv_cache_manager.free(): 已算的物理 KV 块不释放(不回退已算 KV),
            否则会释放该请求全部历史块、违背"已算 KV 不动"、且破坏正在向 D 传输的 KV。
          - 不降级 status、不移出 running: 状态原样保留(保持 RUNNING), 下一拍重排续算。
          - prev_step_scheduled_req_ids.discard: 下拍按"新续算"处理, 避免基类
            _make_cached_request_data 的 assert not scheduled_in_prev_step。
        本拍新分配的尾部块(尚未写入, 回滚在 execute 前发生)保留: 它们不属"已计算 KV",
        下一拍真正执行时写入并复用。FullAttentionManager 对 running 请求的
        get_num_blocks_to_allocate / allocate_new_blocks 在块已足够时均 max(...,0)
        钳到 0, 故保留尾部块不触发负分配或断言; attention 的 causal mask 只读
        num_computed..num_computed+num_new, 不脏读。
        """
        # [DyCP v82] 在 _remove_req_from_output 清 scheduled_new_reqs 之前, 判定
        # 本 req 这拍是否首次调度(在 scheduled_new_reqs 中, 即 worker 尚未注册)。
        was_first_schedule = any(
            r.req_id == req_id for r in output.scheduled_new_reqs
        )
        num_scheduled = output.num_scheduled_tokens.pop(req_id, 0)
        output.total_num_scheduled_tokens -= num_scheduled
        req = self.active_cp_requests.get(req_id)
        if req is not None and req_id in self.requests:
            req.num_computed_tokens = max(
                0, req.num_computed_tokens - num_scheduled
            )
        self._remove_req_from_output(output, req_id)
        self.prev_step_scheduled_req_ids.discard(req_id)
        # [DyCP v82] 首次调度即回滚: worker 从未收到该 req 的 new_req 注册。若按现
        # 行"留 running 续算"处理, 下拍 base 会以 scheduled_cached_reqs 下发 ->
        # worker self.requests[req_id] KeyError(v82 根因)。留 running 原位(保序)并
        # 标 pending, 由 schedule() 末尾 _emit_pending_new_cp_as_new 在下拍子组对齐
        # 时挪到 scheduled_new_reqs 发出(worker 注册)。续算回滚(was_first_schedule=
        # False, worker 已有)不标, 仍走 cached 续算(现行行为不变)。
        if was_first_schedule and req_id in self.requests:
            self._cp_reqs_pending_new_emit.add(req_id)

    # ------------------------------------------------------------------
    # [DyCP 阶段2降级] 执行层 CP 对齐降级(协商已合并进 sync_schedule_confirm)
    # ------------------------------------------------------------------

    def _degrade_cp_from_output(
        self, output: SchedulerOutput
    ) -> bool:
        """[DyCP 阶段2降级] 把本拍 output 中**所有** CP 请求批量 soft-rollback。

        触发时机与 _soft_rollback 的关键区别:
          - _soft_rollback: 共识后**逐个**处理"本端未分到段但 peer 分到了"的 CP 请求
            (min=NOT_SCHEDULED), 是 per-req 的、针对单个请求本端没排上。
          - _degrade_cp_from_output (本方法): 阶段2协商后**本端含长但子组不全含长**
            (local_has_cp=1 且 subgroup_all_schedule_cp_request=0), 是**全量**: 把本端
            本拍排上的**所有** CP 请求一次性剔出 + 回退 num_computed, 对齐子组两端
            num_cp 同 0(D 端不进 dycp all_gather), 避免一端进 hccl all_gather、一端
            不进致不配对 worker hang/全 DP 互锁(v72 取证)。

        两者的底层执行语义**完全一致**(都是由 _rollback_cp_req_from_output 完成
        剔除+回退 num_computed+保留 KV/status/running), 差别只在"触发条件与覆盖范围":
          · soft_rollback: 单个 req, 触发自共识 min=NOT_SCHEDULED;
          · _degrade_cp_from_output: 全部 CP req, 触发自协商 subgroup 不全含长。
        故本方法循环调用公共子方法 _rollback_cp_req_from_output, 不重复实现剔除逻辑。

        sky meta 安全: kv_connector_metadata 由 base schedule() 在 super() 内依
        "调度时刻"已构建; 被降级的 CP 正在 prefill、尚未 finish, 不在 _reqs_need_send
        中, 不在 meta.requests_to_send; cp_req_id 在 super 之后才赋, 故 meta.reqs_in_batch
        恒空。剔除 output 侧 CP 计数不脏读已构建的 meta, 无需重建/补丁。
        返回 True 表示本拍发生了降级(供日志/调用方)。
        """
        degraded_ids: list[str] = []
        for req_id in list(output.num_scheduled_tokens.keys()):
            if req_id not in self.active_cp_requests:
                continue
            self._rollback_cp_req_from_output(output, req_id)
            degraded_ids.append(req_id)
        return len(degraded_ids) > 0

    # ------------------------------------------------------------------
    # [DyCP v82 保序修复] 首次调度被回滚的 CP 请求: 下拍子组对齐时以 new 发出
    # ------------------------------------------------------------------

    def _emit_pending_new_cp_as_new(self, output: SchedulerOutput) -> None:
        """[DyCP v82 保序] 把本拍 scheduled_cached_reqs 里属于
        ``_cp_reqs_pending_new_emit`` 的 CP 请求挪到 scheduled_new_reqs 发出。

        背景: 首次调度被回滚的 CP 请求留 running 原位并标 pending(见
        _rollback_cp_req_from_output)。本拍 base 把它当 running 续算放进
        scheduled_cached_reqs; 若子组已对齐(未被本拍再次回滚, 仍在 cached 中),
        则把它从 cached 挪到 new: worker 走 new 注册路径
        (self.requests[req_id]=CachedRequestState) 而非 cached 查找路径
        (req_state=self.requests[req_id]->KeyError, v82 根因)。
        挪动只动 scheduled_cached_reqs/scheduled_new_reqs, 不动
        num_scheduled_tokens/total/CP 元数据(new/cached 不影响这些; scheduler 侧
        update_from_output 按 num_scheduled_tokens 统一处理, 不区分 new/cached)。
        挪动后清 pending。续算回滚的请求(worker 已有)不进 pending, 仍走 cached 续算。
        NewRequestData 构造与 base scheduler.py:830-847 同款(v2 model runner 带
        prefill_token_ids)。
        """
        if not self._cp_reqs_pending_new_emit:
            return
        cached = output.scheduled_cached_reqs
        # 本拍仍在 cached(被 schedule 且未被本拍回滚)的 pending req
        move_ids = [
            r for r in cached.req_ids if r in self._cp_reqs_pending_new_emit
        ]
        if not move_ids:
            return
        for req_id in move_ids:
            req = self.requests.get(req_id)
            if req is None or req_id not in self.requests:
                # 请求已不在(被 finish/hard_rollback 等清掉), 清 pending
                self._cp_reqs_pending_new_emit.discard(req_id)
                continue
            block_ids = self.kv_cache_manager.get_blocks(req_id).get_block_ids()
            # 从 CachedRequestData 外科式移除(不动 num_scheduled_tokens/total/CP 元数据)
            self._remove_req_from_cached_only(output, req_id)
            # 以 new 发出: worker 注册 + prefill(此时子组已对齐, dycp all_gather 配对)
            if getattr(self, "use_v2_model_runner", False):
                new_req_data = NewRequestData.from_request(
                    req, block_ids, prefill_token_ids=req._all_token_ids
                )
            else:
                new_req_data = NewRequestData.from_request(req, block_ids)
            # [DyCP v85 修复] emit-as-NEW 必须用 "advance 前" 的 num_computed_tokens。
            # 本方法在 schedule() 末尾调用, 此时 base 的 _update_after_schedule 已把
            # req.num_computed_tokens += num_scheduled_tokens 这拍虚推。正常首调 NEW 在
            # _update_after_schedule 之前构造 NewRequestData, 故其带的 num_computed_tokens
            # 是 advance 前(首次=0); 而 emit 在 advance 之后构造, NewRequestData.from_request
            # 会沿用 advance 后的值(如 9)。worker NEW 路径按 num_new = prompt_len -
            # num_computed_tokens 取 token: 带 9 时算出 0 个新 token(全 -1 slot 不写 KV),
            # 而 peer 这拍走 cached-cont 写真实 slot, 两端同一 CP 请求同拍 num_new 不对称
            # -> per-layer dycp all_gather / KV 写入发散 -> 层间集合通信错位死锁(v85)。
            # 这里把虚推 advance 撤回, 取 advance 前值, 使 emit-as-NEW 与正常 fresh-NEW
            # 语义一致。was_first_schedule 的 pending 请求 worker 从未注册、无真实 KV,
            # advance 前值即 fresh 起拍点(正常无 prefix 命中时为 0), 撤回安全。req 侧同步
            # 回退, 待 worker 执行后 update_from_output 重新推进到正确进度。
            num_scheduled_tokens_this_step = output.num_scheduled_tokens.get(req_id, 0)
            num_computed_tokens_before_advance = max(
                0, req.num_computed_tokens - num_scheduled_tokens_this_step
            )
            new_req_data.num_computed_tokens = num_computed_tokens_before_advance
            req.num_computed_tokens = num_computed_tokens_before_advance
            output.scheduled_new_reqs.append(new_req_data)
            self._cp_reqs_pending_new_emit.discard(req_id)

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