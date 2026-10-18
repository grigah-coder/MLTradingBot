"""Lightweight, thread-safe, in-memory TTL cache.

Used to avoid redundant work (network calls to Alpaca News, FinBERT
inference, and technical-indicator maths) within a short time window.

No third-party dependencies: pure standard library so it works with the
existing ``requirements.txt``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

_MISS = object()


class TTLCache:
    """A bounded cache where every entry expires after ``ttl`` seconds.

    * Thread-safe: all mutations happen under a re-entrant lock so it can be
      shared between the ``ThreadPoolExecutor`` workers.
    * Bounded: once ``maxsize`` is reached, expired entries are purged and,
      if still full, the oldest inserted entry is evicted.
    * Monotonic clock: immune to system-clock changes.
    """

    __slots__ = ("ttl", "maxsize", "_store", "_lock")

    def __init__(self, ttl: float, maxsize: int = 1024):
        self.ttl = float(ttl)
        self.maxsize = int(maxsize)
        self._store: Dict[Any, Tuple[float, Any]] = {}
        self._lock = threading.RLock()

    # -- internals -----------------------------------------------------
    def _purge_expired_locked(self, now: float) -> None:
        expired = [k for k, (expires, _) in self._store.items() if expires <= now]
        for key in expired:
            self._store.pop(key, None)

    # -- public API ----------------------------------------------------
    def get(self, key: Any, default: Any = None) -> Any:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return default
            expires, value = item
            if expires <= now:
                self._store.pop(key, None)
                return default
            return value

    def set(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        ttl = self.ttl if ttl is None else float(ttl)
        with self._lock:
            if key not in self._store and len(self._store) >= self.maxsize:
                now = time.monotonic()
                self._purge_expired_locked(now)
                if len(self._store) >= self.maxsize:
                    # Evict the oldest inserted entry (dicts keep insert order).
                    oldest = next(iter(self._store), None)
                    if oldest is not None:
                        self._store.pop(oldest, None)
            self._store[key] = (time.monotonic() + ttl, value)

    def get_or_set(
        self,
        key: Any,
        factory: Callable[[], Any],
        ttl: Optional[float] = None,
    ) -> Any:
        """Return the cached value, or compute+store it with ``factory``.

        The factory may run more than once under heavy concurrency; that is
        an accepted trade-off for not holding the lock during I/O.
        """
        value = self.get(key, _MISS)
        if value is not _MISS:
            return value
        value = factory()
        self.set(key, value, ttl=ttl)
        return value

    def invalidate(self, key: Any) -> None:
        with self._lock:
            self._store.pop(key, None)

    def purge_expired(self) -> None:
        with self._lock:
            self._purge_expired_locked(time.monotonic())

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __contains__(self, key: Any) -> bool:
        return self.get(key, _MISS) is not _MISS

    def __len__(self) -> int:
        return len(self._store)
