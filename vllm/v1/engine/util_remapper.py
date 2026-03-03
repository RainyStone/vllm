# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Dict, Optional
import threading
import asyncio
from vllm.logger import init_logger

logger = init_logger(__name__)
EngineIdentity = bytes

class RequestEngineMapper:
    """Request-Engine Mapper"""

    def __init__(self, ttl_minutes: int = 60, cleanup_interval: int = 60):
        """
        Init

        Args:
            ttl_minutes: time-to-live (minutes)
            cleanup_interval: seconds
        """
        # req_id -> (engine_identity, timestamp)
        self.req_engine_map: Dict[str, tuple[EngineIdentity, float]] = {}

        self.ttl_seconds = ttl_minutes * 60
        self.cleanup_interval = cleanup_interval

        # Lock
        self._lock = threading.RLock() if threading else None
        self._async_lock = asyncio.Lock()

        # flag
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False

        # status
        self.stats = {
            "total_added": 0,
            "total_removed": 0,
            "cleanup_count": 0,
            "active_entries": 0,
        }

    def add_mapping(self, req_id: str, engine: EngineIdentity) -> None:
        timestamp = time.time()

        with self._lock:
            self.req_engine_map[req_id] = (engine, timestamp)

            self.stats["total_added"] += 1
            self.stats["active_entries"] = len(self.req_engine_map)

    def remove_mapping(self, req_id: str) -> Optional[EngineIdentity]:
        with self._lock:
            if req_id not in self.req_engine_map:
                return None

            engine, _ = self.req_engine_map.pop(req_id)

            self.stats["total_removed"] += 1
            self.stats["active_entries"] = len(self.req_engine_map)
            return engine

    def get_engine(self, req_id: str) -> Optional[EngineIdentity]:
        with self._lock:
            if req_id not in self.req_engine_map:
                return None
            engine, timestamp = self.req_engine_map[req_id]

            if time.time() - timestamp > self.ttl_seconds:
                self.remove_mapping(req_id)
                return None

            return engine

    async def start_cleanup_task(self) -> None:
        if self._is_running:
            return

        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self) -> None:
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        while self._is_running:
            await asyncio.sleep(self.cleanup_interval)
            await self.cleanup_expired()

    async def cleanup_expired(self) -> None:
        async with self._async_lock:
            current_time = time.time()
            expired_reqs = []

            for req_id, (engine, timestamp) in self.req_engine_map.items():
                if current_time - timestamp > self.ttl_seconds:
                    expired_reqs.append(req_id)

            for req_id in expired_reqs:
                self.remove_mapping(req_id)

            self.stats["cleanup_count"] += 1

            if expired_reqs:
                logger.info(f"[Cleanup] Removed {len(expired_reqs)} expired entries")

    def get_stats(self) -> dict:
        with self._lock:
            return {
                **self.stats,
                "ttl_seconds": self.ttl_seconds,
                "cleanup_interval": self.cleanup_interval,
                "is_running": self._is_running,
            }

    def clear(self) -> None:
        with self._lock:
            self.req_engine_map.clear()
            self.stats["active_entries"] = 0
