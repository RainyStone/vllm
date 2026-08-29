# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import multiprocessing
import multiprocessing.connection
import time
import weakref

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_tcp_uri, make_zmq_socket
from vllm.utils.system_utils import get_mp_context, set_process_title
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)


class DPCoordinator:
    """Coordinator process used for data-parallel deployments (DP>1).

    Intermediates between multiple DP engine rank processes and one or more
    front-end API server processes.

    * Collects stats from each DP engine (currently just waiting and running
      queue lengths), and publishes these to all front-ends for use in
      load-balancing decisions.

    * Keeps track of the current DP "request wave" number and running state
      of the engines. This is received from the DP rank 0 engine and published
      to the front-end processes along with the current load stats.

      The engines alternate between a global running/paused state. The global
      "request wave" number is a count of the number of times that the workers
      collectively move from a running state to a paused state. This transition
      is synchronized via the all-reduce operation performed in the
      DPEngineCoreProc._has_global_unfinished_reqs method.

    * Broadcasts the START_DP_WAVE message to engines to move them from paused
      to running state when one engine receives a new request. This can happen
      in two cases:
      1) A front-end sending a new request while the engines are paused will
         concurrently notify the coordinator.
      2) An engine receiving a request for a stale request wave while in paused
         state will notify the coordinator.

    Engines will move into running state when receiving a new request or
    START_DP_WAVE message.

    Note that when deployed in External LB mode, no stats will be published by
    the engines and thus updates will only be sent to front-ends when the
    request wave / running state changes.
    """

    def _wait_for_zmq_addrs(self, zmq_addr_pipe) -> tuple[str, str, str]:
        try:
            ready = multiprocessing.connection.wait(
                [zmq_addr_pipe, self.proc.sentinel], timeout=30
            )
            if not ready:
                raise RuntimeError(
                    "DP Coordinator process failed to report ZMQ addresses "
                    "during startup."
                )
            try:
                return zmq_addr_pipe.recv()
            except EOFError:
                raise RuntimeError(
                    "DP Coordinator process failed during startup."
                ) from None
        finally:
            zmq_addr_pipe.close()

    def __init__(
        self, parallel_config: ParallelConfig, enable_wave_coordination: bool = True
    ):
        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "Coordinator only used for data parallel"

        host = parallel_config.data_parallel_master_ip

        # Assume coordinator is colocated with front-end procs when not in
        # either external or hybrid DP LB mode.
        local_only = not parallel_config.local_engines_only
        local_only_eng = dp_size == parallel_config.data_parallel_size_local
        # NOTE(yongji): handling scaling from intra-node to inter-node
        if parallel_config.enable_elastic_ep:
            local_only_eng = False

        def bind_address(local_only: bool) -> str:
            return (
                get_engine_client_zmq_addr(local_only=True, host=host)
                if local_only
                else get_tcp_uri(host, 0)
            )

        front_publish_address = bind_address(local_only)
        back_publish_address = bind_address(local_only_eng)
        back_output_address = bind_address(local_only_eng)

        context = get_mp_context()
        parent_zmq_addr_pipe, child_zmq_addr_pipe = context.Pipe(duplex=False)
        self.proc: multiprocessing.Process = context.Process(
            target=DPCoordinatorProc.run_coordinator,
            name="VLLM_DP_Coordinator",
            kwargs={
                "engine_count": parallel_config.data_parallel_size,
                "front_publish_address": front_publish_address,
                "back_output_address": back_output_address,
                "back_publish_address": back_publish_address,
                "zmq_addr_pipe": child_zmq_addr_pipe,
                "enable_wave_coordination": enable_wave_coordination,
            },
            daemon=True,
        )
        self.proc.start()
        child_zmq_addr_pipe.close()
        (
            front_publish_address,
            back_output_address,
            back_publish_address,
        ) = self._wait_for_zmq_addrs(parent_zmq_addr_pipe)

        self.stats_publish_address = front_publish_address
        self.coord_in_address = back_publish_address
        self.coord_out_address = back_output_address
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])

    def get_stats_publish_address(self) -> str:
        return self.stats_publish_address

    def get_engine_socket_addresses(self) -> tuple[str, str]:
        """Returns tuple of ZMQ input address, output address."""
        return self.coord_in_address, self.coord_out_address

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown coordinator process with configurable timeout."""
        if self._finalizer.detach() is not None:
            shutdown([self.proc], timeout=timeout)


class EngineState:
    def __init__(self):
        self.request_counts = [0, 0]  # [waiting, running]


class DPCoordinatorProc:
    def __init__(
        self,
        engine_count: int,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ):
        set_process_title("DPCoordinator")
        self.ctx = zmq.Context()

        self.engines = [EngineState() for _ in range(engine_count)]

        self.stats_update_interval_ms = min_stats_update_interval_ms
        self.enable_wave_coordination = enable_wave_coordination

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ):
        coordinator = DPCoordinatorProc(
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms,
            enable_wave_coordination=enable_wave_coordination,
        )
        try:
            coordinator.process_input_socket(
                front_publish_address,
                back_output_address,
                back_publish_address,
                zmq_addr_pipe,
            )
        except KeyboardInterrupt:
            logger.info("DP Coordinator process exiting")
        finally:
            if zmq_addr_pipe is not None:
                zmq_addr_pipe.close()

    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
    ):
        decoder = MsgpackDecoder(EngineCoreOutputs)

        # For tracking request wave progression.
        current_wave = 0
        engines_running = False

        # For tracking request counts for internal load-balancing.
        stats_changed = False
        last_stats_step = -1
        last_stats_wave = -1
        last_step_counts: list[list[int]] | None = None

        with (
            make_zmq_socket(
                path=front_publish_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_front,
            make_zmq_socket(
                path=back_output_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as output_back,
            make_zmq_socket(
                path=back_publish_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_back,
        ):
            if zmq_addr_pipe is not None:
                try:
                    zmq_addr_pipe.send(
                        (
                            publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        )
                    )
                finally:
                    zmq_addr_pipe.close()
            # Wait until all engines subscribe.
            for _ in self.engines:
                if publish_back.recv() != b"\x01":
                    logger.error(
                        "DP Coordinator received unexpected message while "
                        "waiting for engines to subscribe"
                    )
                    return
            # Send ready message to engines.
            publish_back.send(b"READY")

            logger.info("All engine subscriptions received by DP coordinator")

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(publish_back, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish_time = 0
            while True:
                elapsed = int(time.time() * 1000) - last_publish_time
                # Send at stats_update_interval_ms interval if the stats have
                # changed, or otherwise every 5 seconds.
                wait_for = self.stats_update_interval_ms if stats_changed else 5000

                # Wait at least 50ms to ensure we've received all stats for
                # the current step.
                min_timeout = 50 if last_step_counts is None else 0

                events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
                if not events:
                    # Poller timeout - publish current stats to front-ends.
                    if last_step_counts is not None:
                        engine_req_counts_list = last_step_counts
                        last_step_counts = None
                    else:
                        engine_req_counts_list = self._get_engine_counts()
                        stats_changed = False

                    to_publish = (engine_req_counts_list, current_wave, engines_running)
                    # [DyCP] Probe(A2): coordinator 心跳 publish 取证。idle 时 cur_wave 是否
                    # 飙升; total_waiting 引擎侧≈0(↔ 193 不在引擎, 在前端回压)。
                    try:
                        _tot_wait = sum((c[0] if c else 0) for c in engine_req_counts_list)
                        _tot_run = sum((c[1] if c else 0) for c in engine_req_counts_list)
                        import logging as _pl
                        _pl.getLogger("vllm.").info(
                            "[DYCP] Probe/coord_pub cur_wave=%s engines_running=%s "
                            "total_waiting=%s total_running=%s n_eng=%s",
                            current_wave, engines_running, _tot_wait, _tot_run,
                            len(engine_req_counts_list) if engine_req_counts_list else 0,
                        )
                    except Exception:
                        pass
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = int(time.time() * 1000)
                    continue

                events = dict(events)
                wave_state_changed = False

                if publish_back in events:
                    buffer = publish_back.recv()
                    if buffer == b"\x01":
                        # NOTE(yongji): newly started engine subscribed
                        # We need to send READY message here instead of receiving
                        # SCALE_ELASTIC_EP notification from engine core client
                        # as SCALE_ELASTIC_EP is only sent when
                        # new engines finished initialization.
                        # Subscription message, on the other hand, is sent
                        # by each engine during initialization
                        publish_back.send(b"READY")
                    elif buffer != b"\x00":
                        logger.error(
                            "DP Coordinator received unexpected message from engines"
                        )

                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer in (b"\x01", b"\x00"):
                        # Ignore subscription messages.
                        continue

                    decoded = msgspec.msgpack.decode(buffer)
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 2
                        and decoded[0] == "SCALE_ELASTIC_EP"
                    ):
                        # Handle scale up notification
                        new_engine_count = decoded[1]
                        current_count = len(self.engines)
                        if new_engine_count > current_count:
                            for _ in range(new_engine_count - current_count):
                                self.engines.append(EngineState())
                            # NOTE(yongji): handle the case
                            # where newly started engines have current_wave = 0
                            # if existing engines just finished a wave
                            # and engine_running isn't updated yet at
                            # CoordinatorProc requests routed to newly started
                            # engines may not wake up existing engines, as long
                            # as 0 < request.wave < existing engines'
                            # current_wave
                            # we note that 0 is the wave number for the new
                            # engine
                            logger.info(
                                "DPCoordinator scaled up from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        else:
                            self.engines = self.engines[:new_engine_count]
                            logger.info(
                                "DPCoordinator scaled down from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        continue  # Skip normal engine notification processing

                    # Wave coordination: handle new-request messages from front-end.
                    # Only process these when wave coordination is enabled
                    if self.enable_wave_coordination:
                        # We received a message on the front-end XPUB socket,
                        # from an API server sending a new request while the
                        # engines are paused, so that we can wake the other
                        # engines.
                        engine_to_exclude, wave = decoded
                        # [DYCP] Probe/wake: coordinator 收到前端新请求(唤醒引擎)。
                        try:
                            logger.info(
                                "[DYCP] Probe/wake-coord_recv engine_to_exclude=%s "
                                "req_wave=%s cur_wave=%s engines_running=%s "
                                "will_start_wave=%s",
                                engine_to_exclude, wave, current_wave,
                                engines_running, (not engines_running),
                            )
                        except Exception:
                            pass
                        if not engines_running:
                            if wave < current_wave:
                                # If the wave number is stale, ensure the message
                                # is handled by all the engines.
                                engine_to_exclude = None

                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(
                                publish_back, current_wave, engine_to_exclude
                            )

                if output_back in events:
                    # We received a message from one of the engines.

                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert not outputs.outputs
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index
                    scheduler_stats = outputs.scheduler_stats
                    if scheduler_stats:
                        # 1. Updated request load stats - update our local
                        # state with these.
                        stats = self.engines[eng_index].request_counts
                        stats_step = scheduler_stats.step_counter
                        stats_wave = scheduler_stats.current_wave
                        if (
                            stats_wave > last_stats_wave
                            or stats_wave == last_stats_wave
                            and stats_step > last_stats_step
                        ):
                            if stats_changed:
                                last_step_counts = self._get_engine_counts(do_copy=True)
                            last_stats_step = stats_step
                            last_stats_wave = stats_wave
                        elif stats_wave != last_stats_wave or (
                            stats_step != last_stats_step
                        ):
                            logger.warning(
                                "Received stats for out-of-order "
                                "step (%d, %d) from engine %d (expected "
                                "> (%d, %d))",
                                stats_wave,
                                stats_step,
                                eng_index,
                                last_stats_wave,
                                last_stats_step,
                            )
                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs
                        stats_changed = True

                    # Wave coordination: handle wave completion and start notifications
                    # Only process these when wave coordination is enabled
                    if self.enable_wave_coordination:
                        if (wave := outputs.wave_complete) is not None:
                            # 2. Notification from rank 0 engine that we've
                            # moved into the global paused state
                            # (engines_running==False).
                            if current_wave <= wave:
                                new_wave = wave + 1
                                # [DYCP] Probe/wake: coordinator 收到 rank0 的 wave_complete, 引擎暂停。
                                logger.info(
                                    "[DYCP] Probe/wake-coord_wave_complete old_wave=%s "
                                    "new_wave=%s engines_running=%s->False eng_index=%s",
                                    current_wave, new_wave, engines_running, eng_index,
                                )
                                # [DyCP] Probe(A2): wave_complete 时引擎侧 waiting 总数 +
                                # 是否 idle 推进(racing): total_waiting==0 却仍在涨 wave =
                                # idle 跑飞经 coordinator 反向传染。
                                try:
                                    _tw = sum(
                                        (e.request_counts[0] if e.request_counts else 0)
                                        for e in self.engines
                                    )
                                    import logging as _pl
                                    _pl.getLogger("vllm.").info(
                                        "[DYCP] Probe/coord_wave_complete_extra old_wave=%s "
                                        "new_wave=%s total_waiting=%s racing=%s",
                                        current_wave, new_wave, _tw, (_tw == 0),
                                    )
                                except Exception:
                                    pass
                                current_wave = new_wave
                                engines_running = False
                                wave_state_changed = True
                        elif (wave := outputs.start_wave) is not None and (
                            wave > current_wave
                            or (wave == current_wave and not engines_running)
                        ):
                            # 3. The engine received request for a non-current wave
                            # so we must ensure that other engines progress to the
                            # next wave (race condition handling).
                            logger.info(
                                "[DYCP] Probe/wake-coord_start_wave_from_eng wave=%s "
                                "eng_index=%s engines_running=%s->True",
                                wave, eng_index, engines_running,
                            )
                            current_wave = wave
                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(publish_back, wave, eng_index)

                if wave_state_changed:
                    message = (None, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(message))

    @staticmethod
    def _send_start_wave(
        socket: zmq.Socket, wave: int, exclude_engine_index: int | None
    ):
        """Broadcast the START_DP_WAVE message to all the engines.
        It includes the current wave number and index of engine which
        has already received a request with this wave number and so doesn't
        require additional notification.
        """
        # [DYCP] Probe/wake: coordinator 广播 START_DP_WAVE 唤醒引擎。
        try:
            logger.info(
                "[DYCP] Probe/wake-coord_send_start_wave wave=%s exclude_engine_index=%s",
                wave, exclude_engine_index,
            )
        except Exception:
            pass
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart((EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))

    def _get_engine_counts(self, do_copy=False) -> list[list[int]]:
        """Return list of [waiting, running] count lists for each engine."""
        if do_copy:
            return [copy.copy(e.request_counts) for e in self.engines]
        return [e.request_counts for e in self.engines]
