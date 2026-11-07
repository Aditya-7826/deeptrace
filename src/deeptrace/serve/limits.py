"""In-memory LRU cache and per-IP sliding-window rate limiter (single worker process)."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Any


class LRUCache:
    def __init__(self, max_items: int = 128) -> None:
        self.max_items = max_items
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class RateLimiter:
    """At most `limit` requests per `window_s` seconds per client IP."""

    def __init__(self, limit: int = 60, window_s: float = 60.0, clock=time.monotonic) -> None:
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client: str) -> float | None:
        """None if allowed (and counted), else seconds until the next request is allowed."""
        now = self.clock()
        with self._lock:
            q = self._hits.setdefault(client, deque())
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return max(self.window_s - (now - q[0]), 0.0)
            q.append(now)
            if len(self._hits) > 10_000:  # drop idle clients so memory stays bounded
                for ip in [ip for ip, d in self._hits.items() if not d]:
                    del self._hits[ip]
            return None
