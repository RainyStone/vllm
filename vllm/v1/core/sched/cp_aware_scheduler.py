# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CP-Aware Scheduler for Dynamic Context Parallel.

Extends the existing Scheduler with CP awareness while preserving the
per-DP independent process architecture. CP requests are coordinated
via a post-schedule all_gather_object consensus protocol that aligns
per-request scheduling states by request_id (not by fixed slot position).

Design rationale: CP requests are routed to the DyCP subgroup that owns them,
so only the cooperating ranks need to agree, and no pre-schedule announce/vote
phase is needed. Coordination is a single post-schedule all_gather_object that
exchanges each rank's {req_id: status} dict; per-request decisions are merged
by req_id MIN, agreeing on whether each rank successfully scheduled each CP
request in the current step.
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

# Three-state encoding for post-schedule consensus.
SCHEDULED = 2
NOT_SCHEDULED = 1
PREEMPTED = 0

# all_gather_object 载荷中的保留键: 本端本拍实际调度到的每个 CP 长请求的 token 范围,
# value=[num_scheduled_tokens, num_computed_tokens](均为 advance 后值, 范围定义
# [num_computed - num_scheduled, num_computed))。与 __scheduled_cp__ 并列, 在收集
# req_id 并集、本地/接收探针、状态 MIN 合并处一律跳过此两个保留键, 不当作 req_id。
_RESERVED_TOKEN_RANGES_KEY = "__token_ranges__"


class CPSyncProtocol:
    """Post-schedule consensus protocol for CP request scheduling.

    Uses the DyCP subgroup (``dycp_group``) gloo all_gather_object to
    exchange each rank's {req_id: status} dict, then merges per-request states
    by req_id MIN to agree on whether each active CP request was successfully
    scheduled on ALL ranks of its CP group in the current step.

    Three-state encoding per request:
        SCHEDULED (2)     - this rank scheduled the request
        NOT_SCHEDULED (1) - this rank did not schedule it (budget exhausted)
        PREEMPTED (0)     - this rank preempted it (KV cache eviction)

    After per-req_id MIN merge:

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
        # rank and produced duplicate outputs.
        self.dycp_group = dycp_group
        self.cp_world_size = cp_world_size
        self.cp_rank = cp_rank

    def sync_schedule_confirm(
        self,
        active_ids: list[str],
        status: list[int],
        local_scheduled_cp: int,
        token_ranges: dict[str, tuple[int, int]],
    ) -> tuple[list[str], list[str], list[str], int, dict[str, int]]:
        """Post-schedule consensus using all_gather_object by req_id, 合并阶段2协商.

        用子组内一次 gloo all_gather_object 交换 {req_id: status} 字典, 按 req_id
        精确合并, 取代原固定槽位 tensor + all_reduce MIN。

        根因(本方法替代的旧机制缺陷): 旧机制用固定槽位 tensor + all_reduce MIN,
        对齐完全依赖各 rank sorted(active_ids) 同 position = 同 req_id。但
        add_request_async 子组内串行 await send 造成 owner 与非 owner 收到 add 的
        到达窗口; 窗口内两端 active 集合不一致 -> sorted 后同 position 不是同一
        req_id -> 共识把不同 req 的状态投到同一槽 MIN -> position 错位(不打注入也
        偶发 v78/v82 类)。本方法按 req_id 合并, 不依赖 position, 从根消除错位;
        并集无截断、不要求各端同 N、不要求同 key 集合。

        Args:
            active_ids: Active CP request IDs (sorted)。object 版仅作本端遍历用,
                        合并以 all_gather 后的并集为准, 不再要求跨 rank identical。
            status: Per-request status on this rank
                    (SCHEDULED=2 / NOT_SCHEDULED=1 / PREEMPTED=0), 与 active_ids 对齐。
            local_scheduled_cp: 本拍本 rank 是否"调度到" CP 长请求(1=是, 0=否)。
                                注意是"本步是否调度到", 非"是否持有"——active 有但本步
                                未调度到=0。写入保留键 __scheduled_cp__, all_gather 后
                                取全子组 MIN 即 subgroup_all_schedule_cp_request。
            token_ranges: req_id -> (num_scheduled_tokens, num_computed_tokens), 仅含
                          本端本拍实际调度到(出现在 output.num_scheduled_tokens)的 CP 长
                          请求。num_computed_tokens 为 advance 后值(理由见下方不变量)。
                          写入保留键 __token_ranges__, 仅取证(探针打印各端 token 范围),
                          不影响三态合并/回滚/降级决策。本拍无实际计算(finished-report-
                          SCHEDULED 等)的请求不进此 dict, 范围留空。

        Returns:
            (confirmed_ids, soft_rollback_ids, hard_rollback_ids,
             subgroup_all_schedule_cp_request) —— 返回合约与旧机制完全一致, 下游
             (_degrade_cp_from_output / _soft_rollback / _hard_rollback / dummy 门控)
             零改动。其中 subgroup_all_schedule_cp_request = 子组所有端
             local_scheduled_cp 的 MIN: =1 当且仅当子组所有端本拍都调度到 CP 长请求;
             =0 表示至少一端本拍没调度到(含长端须降级对齐)。

        缺 key 语义: 某 req 不在本端发来的 dict 中(未收到 add / 已 finish 清出),
        合并时一律按 NOT_SCHEDULED 取("本端没排到它"), 不误判成 PREEMPTED(仅显式投
        PREEMPTED 才算 hard_rollback 危险值)。故 add 异步到达时一端 SCHEDULED / 另一端
        没到 -> MIN=NOT_SCHEDULED -> soft_rollback, 与"peer 没排到它"语义一致。

        token 范围取样点不变量(关键): 本方法在 CPAwareScheduler.schedule() 中于
        output = super().schedule() 返回后调用, 而 base 的 _update_after_schedule
        (request.num_computed_tokens += num_scheduled_token) 已在 super() 内部 return
        前执行, 故此处各 req 的 num_computed_tokens 已是 advance 后值 = comp_before +
        sched。因此本拍真正计算的 token 范围 = [num_computed - num_scheduled, num_computed)
        (即 range_start = num_computed_tokens - num_scheduled_tokens, range_end =
        num_computed_tokens)。禁止误用 [num_computed, num_computed + num_scheduled):
        该式仅在 advance 之前(= comp_before)成立, 在本取样点会把范围整体平移一个 sched、
        误报成下一步起点。__token_ranges__ 仅用于探针取证, 不参与决策。
        """
        # 全程计时(探针): t_start 整个共识起点; 三段时长 = gather / local / total。


        # 构造本端要发出的 dict: __scheduled_cp__ 是保留键(代替旧槽位机制的标志槽),
        # 表示本 rank 本步是否调度到长请求(1/0); 其余 key=req_id, value=该 req 本端状态。
        local_status_dict: dict = {"__scheduled_cp__": local_scheduled_cp}
        for i, req_id in enumerate(active_ids):
            local_status_dict[req_id] = status[i]

        # 保留键 __token_ranges__: 携带本端本拍实际调度到的每个 CP 长请求的 token
        # 范围, value=[num_scheduled_tokens, num_computed_tokens](均为 advance 后值,
        # 范围 [num_computed - num_scheduled, num_computed) 见方法 docstring 不变量)。
        # 各 rank 都写该键(无调度 CP 时为空 dict)以保证保留键集合一致; 收集 req_id
        # 并集、本地/接收探针、状态 MIN 合并处一律跳过 __scheduled_cp__ 与
        # __token_ranges__ 两个保留键, 不当作 req_id 处理。
        token_ranges_payload: dict[str, list[int]] = {}
        for req_id, range_value in token_ranges.items():
            num_scheduled_tokens_payload = range_value[0]
            num_computed_tokens_payload = range_value[1]
            token_ranges_payload[req_id] = [
                int(num_scheduled_tokens_payload),
                int(num_computed_tokens_payload),
            ]
        local_status_dict[_RESERVED_TOKEN_RANGES_KEY] = token_ranges_payload

        # 子组内一次 gloo all_gather_object。返回 gathered_objects[src_cp_rank] =
        # 子组内 cp_rank=src 的端发来的 dict, 长度恒=子组端数(dycp_size, 可 >2);
        # 各段天然按 cp_rank 区分来源。空拍端 dict={"__scheduled_cp__":0} 仍参与
        # gather(旧机制空拍不进共识致忙端 hang 的根因已由此根治), 形状不对齐安全。
        gathered_objects: list[dict] = [
            {} for _ in range(self.cp_world_size)
        ]
        torch.distributed.all_gather_object(
            gathered_objects, local_status_dict, group=self.dycp_group,
        )

        # 子组所有端 scheduled_cp 的 MIN -> subgroup_all_schedule_cp_request。
        # 普通循环取 min, 不用生成器表达式。
        subgroup_all_schedule_cp_request = SCHEDULED  # 初始为最大值, 下面循环取 min
        for src_rank in range(self.cp_world_size):
            value = int(gathered_objects[src_rank].get("__scheduled_cp__", 0))
            if value < subgroup_all_schedule_cp_request:
                subgroup_all_schedule_cp_request = value

        # 收集所有 req_id 并集(sorted 稳定, 仅影响日志/列表顺序, 不影响语义与下游)。
        # 普通循环构造 set 再 sorted, 不用集合推导式。
        req_ids_union_set = set()
        for src_rank in range(self.cp_world_size):
            for key in gathered_objects[src_rank].keys():
                if key in ("__scheduled_cp__", _RESERVED_TOKEN_RANGES_KEY):
                    continue
                req_ids_union_set.add(key)
        all_req_ids = sorted(req_ids_union_set)

        # 按 req_id 精确合并: 各端 status 取 MIN(缺 key -> NOT_SCHEDULED), 三态分类。
        # 普通循环取 min, 不用生成器表达式; 不引入探针变量。
        confirmed: list[str] = []
        soft_rollback: list[str] = []
        hard_rollback: list[str] = []
        # confirmed_token_alignment: req_id -> min_num_computed_tokens(各 rank advance
        # 后 num_computed_tokens 的最小值, 即子组共识的 ENDoMin)。仅含 confirmed 且
        # 对称(req 在子组每个 rank 的 __token_ranges__ 中都有条目, 各 rank 本拍都真
        # 调度了它)的长请求。供 scheduler 端 _align_cp_token_range 把大端 ENDo 削到此处,
        # 使各 rank 对同一 confirmed 长请求切同一 total, 消除 per-layer dycp all_gather
        # 形状不配对的死锁根因。详见 _align_cp_token_range docstring。
        confirmed_token_alignment: dict[str, int] = {}
        for req_id in all_req_ids:
            min_status = SCHEDULED  # 初始最大值, 下面循环取 min
            for src_rank in range(self.cp_world_size):
                value = int(gathered_objects[src_rank].get(req_id, NOT_SCHEDULED))
                if value < min_status:
                    min_status = value
            if min_status >= SCHEDULED:
                confirmed.append(req_id)
            elif min_status >= NOT_SCHEDULED:
                soft_rollback.append(req_id)
            else:
                hard_rollback.append(req_id)
            # ---- 对齐目标计算: 仅 confirmed 且对称(各 rank __token_ranges__ 都有)的 req ----
            if min_status < SCHEDULED:
                continue
            # 收集各 rank 对该 req 的 token 范围(num_scheduled, num_computed), 仅取各 rank
            # __token_ranges__ 中真有该 req 的(finished-report-SCHEDULED 的非对称端不在
            # __token_ranges__, 跳出对称要求、不进对齐, 留给既有 finished/dummy 路径)。
            per_rank_ends = []  # 各 rank 的 (range_start, num_computed_tokens) 列表
            symmetric = True    # 子组每个 rank 的 __token_ranges__ 都有该 req
            for src_rank in range(self.cp_world_size):
                per_req_ranges = gathered_objects[src_rank].get(
                    _RESERVED_TOKEN_RANGES_KEY, {}
                )
                entry = per_req_ranges.get(req_id)
                if entry is None:
                    symmetric = False
                    break
                num_scheduled_tokens_local = int(entry[0])
                num_computed_tokens_local = int(entry[1])
                range_start_local = (
                    num_computed_tokens_local - num_scheduled_tokens_local
                )
                per_rank_ends.append(
                    (range_start_local, num_computed_tokens_local)
                )
            if not symmetric or not per_rank_ends:
                # 非对称: 某 rank finished-report-SCHEDULED 或本拍没真调度 -> 不对齐,
                # 留给既有 finished/dummy 路径处理。注释明确 out-of-scope。
                continue
            # assert start 各 rank 一致: lockstep 不变量。confirmed 对称 req 的 start
            # 由对称共识保证各 rank 相等(soft/hard/degrade 对称返回同一列表 -> comp_before
            # 同步推进; 各 rank 都排到该 req 即各 rank comp_before 相同)。若 start 不一致
            # 则锁步已破裂, 不在本计划处理范围 -> assert 报错暴露, 切勿静默。
            first_range_start = per_rank_ends[0][0]
            for (range_start_local, _) in per_rank_ends:
                assert range_start_local == first_range_start, (
                    "[DyCP align] confirmed req %s 各 rank range_start 不一致: "
                    "%s (lockstep 破裂, 当前仅处理 start 一致情况, 见计划文档)"
                    % (req_id, per_rank_ends)
                )
            # min_end = 各 rank num_computed_tokens(advance后) 的最小值。因 start 一致,
            # 削 end 到 min_end 等价于削 count 到 min_count, 各 rank 对同一 total 切互补分片。
            min_num_computed_tokens = min(
                num_computed_tokens_local
                for (_, num_computed_tokens_local) in per_rank_ends
            )
            confirmed_token_alignment[req_id] = min_num_computed_tokens


        # local + gather = total 恒等(用派生式保证), 可自检对账。留工作区不提交。

        if soft_rollback or hard_rollback:
            logger.info(
                "[DYCP] CP confirm: confirmed=%d soft_rollback=%d hard_rollback=%d"
                " on rank %d",
                len(confirmed),
                len(soft_rollback),
                len(hard_rollback),
                self.cp_rank,
            )

        return (
            confirmed,
            soft_rollback,
            hard_rollback,
            subgroup_all_schedule_cp_request,
            confirmed_token_alignment,
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
    - Post-schedule all_gather_object (by req_id) agrees on confirmed/rollback decisions
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
        # Active CP requests require this rank to participate in the all_gather_object
        # consensus even when it has no locally-scheduled tokens, so treat them as
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
          3. all_gather_object by req_id — agree on confirmed / rollback decisions
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

        # 本拍本 rank 是否"调度到" CP 长请求(1=是, 0=否; 非"是否持有")。
        # 用于合并共识 sync_schedule_confirm 的保留键 __scheduled_cp__: all_gather
        # 后子组所有端该值的 MIN = subgroup_all_schedule_cp_request(子组是否所有端
        # 本拍都调度到长), 驱动下方阶段2降级。须在 CP 元数据标注(num_cp_request 已赋)后计算。
        local_scheduled_cp = 1 if output.num_cp_request > 0 else 0

        # No sync needed (single DP or test environment without dp_group).
        if self.cp_sync is None:
            self._preempted_this_step.clear()
            return output

        # [DyCP] 合并修复(v81 根因): 不再因"本端无 active_cp"就提前 return。
        # 空拍也必须进 sync_schedule_confirm 的 all_gather_object: 否则忙端
        # (有 active_cp)的共识会一直等空端 -> hang。原 tensor 固定槽位机制下曾
        # 因"空拍跳过共识、忙端调共识"配合 align 的独立 all_reduce 形状错配抢
        # 同一 dycp_group 致 gloo Connection reset 崩(v81)。现 object 版合并成
        # 一次 all_gather_object, 空拍端 dict={"__scheduled_cp__":0} 仍参与
        # gather, 与忙端同调用配对(不要求 key 集合一致), 根治 hang 与崩。
        # Build per-request status vector for the all_gather_object.
        active_ids = sorted(self.active_cp_requests.keys())
        status: list[int] = []
        finished_ids: list[str] = []
        # token_ranges: req_id -> (num_scheduled_tokens, num_computed_tokens)。仅含本端
        # 本拍实际调度到的 CP 长请求(出现在 output.num_scheduled_tokens); finished-report
        # SCHEDULED 的请求不在 output.num_scheduled_tokens, 不进此 dict(本拍无实际计算)。
        # num_computed_tokens 取 advance 后值, 范围见 sync_schedule_confirm docstring 不变量。
        token_ranges: dict[str, tuple[int, int]] = {}
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
                # token 范围取证(post-advance): 本拍真正计算区间
                # [num_computed_tokens - num_scheduled_tokens, num_computed_tokens)。
                token_ranges[req_id] = (
                    int(output.num_scheduled_tokens[req_id]),
                    int(req.num_computed_tokens),
                )
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

        confirmed, soft_rollback_ids, hard_rollback_ids, subgroup_all_schedule_cp_request, confirmed_token_alignment = (
            self.cp_sync.sync_schedule_confirm(
                active_ids, status, local_scheduled_cp, token_ranges
            )
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

        # [DyCP 阶段2] 协商降级: 本端含长(local_scheduled_cp=1)但子组不全含长
        # (subgroup_all_schedule_cp_request=0) -> 全量剔除本拍 CP 请求并回退
        # num_computed(降级), 对齐子组两端 num_cp 同 0(都不进 dycp all_gather)。
        # 降级插在 soft/hard_rollback 之前且互斥安全: 降级场景下至少一端空拍
        # (其 scheduled_cp=0、dict 不含该 CP req, 合并取 NOT_SCHEDULED), 该端不会投 PREEMPTED, 故
        # hard_rollback_ids 为空; soft_rollback_ids 含本端全部 cp, 但已被降级从
        # output 剔除(num_scheduled_tokens 已 pop), _soft_rollback 的 SCHEDULED
        # 分支判 not in num_scheduled_tokens 走 else 无害空转, 不重复回退。
        if local_scheduled_cp == 1 and subgroup_all_schedule_cp_request == 0:
            self._degrade_cp_from_output(output)
        if soft_rollback_ids:
            output = self._soft_rollback(output, soft_rollback_ids)
        if hard_rollback_ids:
            output = self._hard_rollback(output, hard_rollback_ids)

        # [DyCP token范围对齐] 共识后把 confirmed 长请求的 token 范围(END)削到子组 MIN,
        # 使各 rank 对同一 confirmed 长请求切同一 total, 消除 per-layer dycp all_gather
        # 形状不配对的死锁根因。仅处理 confirmed 对称 req(各 rank 均 SCHEDULED 且本拍都
        # 真调度), 与 degrade/soft/hard 互斥(those 不作用于 confirmed req), 故插在其后
        # 安全。详见 _align_cp_token_range docstring。执行顺序: degrade -> soft -> hard
        # -> align -> _emit_pending_new_cp_as_new。
        if confirmed_token_alignment:
            self._align_cp_token_range(output, confirmed_token_alignment)

        self._preempted_this_step.clear()

        # 侧是否对本拍 prefill 请求切 chunk。切 chunk 判据(worker mla_cp.py
        # generate_dp_chunked_metadata): context_len = seq_lens - query_lens = 本拍
        # 调度前已算的 tokens(advance 前的 num_computed_tokens); prefill 续算且
        # context_len>0 时, 若 workspace 分摊后的 max_context_chunk<max_context_len 则
        # num_chunks=cdiv(max_context_len,max_context_chunk)>1 即切 chunk。本处只取
        # scheduler 侧能拿到的原生量(不反向依赖 worker 的 workspace 配置去预测确切
        # num_chunks, 避免口径不一致误导); 真正切了与否的权威判定看 worker 侧
        # context_len = req.num_computed_tokens(advance 后) - num_scheduled_tokens(本拍),
        # 差即 advance 前已算(与 worker 同义)。此处在 align 后 emit 前, align 只削
        # 本拍多算的(不动已算 comp_before), emit 还未撤回首调, 故该差值正确。
        # prefill 续算判据: 0 < context_len < num_prompt_tokens(已算部分但未算完 prompt)。
        # decode(context_len>=num_prompt_tokens) 与 首 prefill(context_len=0) 不触发 chunk。
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
    
    def _align_cp_token_range(
        self,
        output: SchedulerOutput,
        confirmed_token_alignment: dict[str, int],
    ) -> None:
        """共识后把 confirmed 长请求的 token 范围(END)削到子组 MIN, 对齐各 rank 切分 total。

        触发: 某长请求在子组各 rank 均 SCHEDULED(confirmed), 但各 rank 本轮调度的
        token 数(count / num_scheduled_tokens)因 budget/短请求负载不同而不一致。
        worker 侧 PCPManager._get_local_cp_tokens 把该请求 total 在 cp_world_size
        个 rank 间均分, 各 rank 对同一 total 切互补分片; per-layer dycp all_gather
        要求各 rank 形状一致 -> 各 rank 必须对同一 CP 请求调度相同 count。count 不一致
        则 total 不同 -> 切分不互补 -> per-layer all_gather 形状不配对 -> 层间错位死锁。
        本方法把较大端的 ENDo 削到子组 min_end, 使各 rank 切同一 total, 根治此死锁。

        仅处理 confirmed 且对称的请求(min==SCHEDULED 且子组每个 rank 的 __token_ranges__
        都有该 req, 由 sync_schedule_confirm 的 confirmed_token_alignment 过滤保证)。
        非对称(含 finished-report-SCHEDULED)不在本方法范围, 留给既有 finished/dummy 路径。

        start 一致性: confirmed 对称 req 的 start(= num_computed_tokens -
        num_scheduled_tokens, 两者 advance 后值)由对称共识保证各 rank 相等, sync 层已
        assert 校验; 本方法据此把 ENDo 削到 min_end(等价削 count 到 min_count)。

        语义(与 soft_rollback "留块+回退 advance" 同源, 但 req 仍留在 output 仅变小,
        不像 soft/degrade 全量剔出):
          - output.num_scheduled_tokens[req]: local_count -> min_count(= local_end - min_end)。
          - output.total_num_scheduled_tokens: 扣减多算的 alignment_reduction_tokens。
          - req.num_computed_tokens(advance 后 = local_end): 回退到 min_end(撤回 advance
            多推部分, 保留之前拍真实进度; 与 soft_rollback 回退同源)。
          - 不改 cp_rank_scheduled_tokens/cp_req_id/req_id_to_cp_size/num_cp_request:
            req 仍是 CP 请求、仍在 output, 只是变小(区别 soft/degrade 把 req 整体剔出)。
          - 不 kv_cache_manager.free: 本拍新分配尾部块未写入、下拍复用(留块原则),
            FullAttentionManager get_num_blocks_to_allocate 钳 0, 无负分配/断言;
            causal mask 只读 [start, start+min_count), 不脏读。
          - 每请求数据(CachedRequestData/NewRequestData 的 num_computed_tokens = START,
            advance 前值; count 由 num_scheduled_tokens 决定): START 字段不动、天然正确,
            仅 count 经 num_scheduled_tokens 改动而正确。PP new_token_ids(async DyCP 下为
            空)若非空则防御性截到 min_count。
        """
        for req_id, min_num_computed_tokens in confirmed_token_alignment.items():
            # 仅当 req 仍在本次 output(未被 degrade/soft/hard 剔除)才对齐;
            # 已被剔除的 req 不在 output.num_scheduled_tokens, 跳过。
            if req_id not in output.num_scheduled_tokens:
                continue
            req = self.active_cp_requests.get(req_id)
            if req is None or req_id not in self.requests:
                # 应不发生(confirmed 对称 req 各 rank 都排到、未 finish); 防御性跳过。
                continue
            local_num_scheduled_tokens = int(output.num_scheduled_tokens[req_id])
            local_num_computed_tokens = int(req.num_computed_tokens)
            # 本端 ENDo(local_num_computed_tokens) 已是/低于子组 MIN -> 无需削减, no-op。
            if local_num_computed_tokens <= min_num_computed_tokens:
                continue
            # 因 start 各 rank 一致(已 assert), reduction = local_end - min_end =
            # local_count - min_count, 削 ENDo 到 min_end 即削 count 到 min_count。
            alignment_reduction_tokens = (
                local_num_computed_tokens - min_num_computed_tokens
            )
            aligned_num_scheduled_tokens = (
                local_num_scheduled_tokens - alignment_reduction_tokens
            )
            # 1) output 侧 count: 削到 min_count。
            output.num_scheduled_tokens[req_id] = aligned_num_scheduled_tokens
            output.total_num_scheduled_tokens -= alignment_reduction_tokens
            # 2) req 侧 num_computed_tokens(advance 后 = local_end): 回退到 min_end,
            #    撤回 advance 多推部分, 保留之前拍真实进度。
            req.num_computed_tokens = min_num_computed_tokens
            # 3) PP new_token_ids 防御性截到 min_count(async DyCP 下该列表为空, no-op;
            #    若未来关 async 则截断保证 PP 下发的 token 数与对齐后 count 一致)。
            cached_reqs = output.scheduled_cached_reqs
            if cached_reqs is not None and req_id in cached_reqs.req_ids:
                cached_idx = cached_reqs.req_ids.index(req_id)
                if (
                    cached_reqs.new_token_ids
                    and cached_idx < len(cached_reqs.new_token_ids)
                    and len(cached_reqs.new_token_ids[cached_idx])
                    > aligned_num_scheduled_tokens
                ):
                    cached_reqs.new_token_ids[cached_idx] = (
                        cached_reqs.new_token_ids[cached_idx][
                            :aligned_num_scheduled_tokens
                        ]
                    )
            # 不动 cp 标志元数据; 不 free KV。探针记录对齐动作。

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
            (local_scheduled_cp=1 且 subgroup_all_schedule_cp_request=0), 是**全量**: 把本端
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
            req = self.active_cp_requests.get(req_id)
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
        # next step (one rank finishes a step earlier), causing req_id-set
        # divergence between ranks; per-req_id merge tolerates this (missing key
        # = NOT_SCHEDULED) but finishing here would still let the request linger
        # on the slower rank via spurious rollback, so cleanup stays in schedule().

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