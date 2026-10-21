"""In-memory tracking of live bracket legs (stop-loss / take-profit).

Alpaca creates bracket children *after* the entry is accepted, and a replace
request returns a **new** order id. Keeping a local map of the currently live
leg ids means the bot can:

* cancel/replace the right order instead of guessing,
* forget an id the moment the broker reports it as filled, cancelled or
  expired (no endless retries against a closed order),
* tell an automated OCO/bracket cancellation apart from a genuine
  user/strategy cancellation when deciding whether to alert.

The module is dependency-free and thread-safe because broker callbacks and the
trading loop may touch it concurrently.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, List, Optional

STOP_LOSS = "stop_loss"
TAKE_PROFIT = "take_profit"
_PROTECTIVE_KINDS = (STOP_LOSS, TAKE_PROFIT)


class OrderTracker:
    """Thread-safe ``order id -> (symbol, leg)`` index."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._orders: Dict[str, Dict[str, str]] = {}
        self._by_symbol: Dict[str, Dict[str, str]] = {}

    # -- internals -----------------------------------------------------
    def _forget_locked(self, order_id: str) -> None:
        info = self._orders.pop(order_id, None)
        if info is None:
            return
        legs = self._by_symbol.get(info["symbol"])
        if not legs:
            return
        for kind, tracked_id in list(legs.items()):
            if tracked_id == order_id:
                legs.pop(kind, None)
        if not legs:
            self._by_symbol.pop(info["symbol"], None)

    def _register_locked(
        self, order_id: str, symbol: str, kind: str, status: str = ""
    ) -> None:
        if kind not in _PROTECTIVE_KINDS:
            return
        self._forget_locked(order_id)
        # Only one leg of each kind is live per symbol; drop the previous id
        # so a replace request cannot leave the old order tracked forever.
        previous = self._by_symbol.get(symbol, {}).get(kind)
        if previous and previous != order_id:
            self._forget_locked(previous)
        self._orders[order_id] = {
            "symbol": symbol,
            "kind": kind,
            "status": status,
        }
        self._by_symbol.setdefault(symbol, {})[kind] = order_id

    # -- public API ----------------------------------------------------
    def register(
        self,
        order_id: Any,
        symbol: Any,
        kind: str,
        status: Any = "",
    ) -> None:
        """Track ``order_id`` as the live ``kind`` leg for ``symbol``."""
        oid = str(order_id or "").strip()
        sym = str(symbol or "").strip().upper()
        if not oid or not sym:
            return
        with self._lock:
            self._register_locked(oid, sym, kind, str(status or "").lower())

    def sync(self, live_orders: Iterable[Dict[str, Any]]) -> List[str]:
        """Replace the tracker with the broker's live bracket legs.

        ``live_orders`` is an iterable of normalised dicts with ``id``,
        ``symbol``, ``kind`` and optional ``status``. Any id that is no longer
        live (filled, cancelled or expired) is purged and returned so the
        caller can log it.
        """
        live: Dict[str, Dict[str, str]] = {}
        for item in live_orders or ():
            oid = str(item.get("id") or "").strip()
            sym = str(item.get("symbol") or "").strip().upper()
            kind = str(item.get("kind") or "")
            if not oid or not sym or kind not in _PROTECTIVE_KINDS:
                continue
            live[oid] = {
                "symbol": sym,
                "kind": kind,
                "status": str(item.get("status") or "").lower(),
            }

        with self._lock:
            stale = [oid for oid in self._orders if oid not in live]
            self._orders.clear()
            self._by_symbol.clear()
            for oid, info in live.items():
                self._register_locked(
                    oid, info["symbol"], info["kind"], info["status"]
                )
        return stale

    def forget(self, order_id: Any) -> None:
        """Drop a single order id (after a cancel, fill or expiry)."""
        oid = str(order_id or "").strip()
        if not oid:
            return
        with self._lock:
            self._forget_locked(oid)

    def forget_symbol(self, symbol: Any) -> None:
        """Drop every leg for ``symbol`` (position closed)."""
        sym = str(symbol or "").strip().upper()
        if not sym:
            return
        with self._lock:
            for kind, oid in list(self._by_symbol.get(sym, {}).items()):
                self._orders.pop(oid, None)
                self._by_symbol.get(sym, {}).pop(kind, None)
            self._by_symbol.pop(sym, None)

    def clear(self) -> None:
        with self._lock:
            self._orders.clear()
            self._by_symbol.clear()

    def info(self, order_id: Any) -> Optional[Dict[str, str]]:
        """Return ``{"symbol", "kind", "status"}`` for a tracked id."""
        oid = str(order_id or "").strip()
        if not oid:
            return None
        with self._lock:
            info = self._orders.get(oid)
            return dict(info) if info else None

    def order_id(self, symbol: Any, kind: str) -> Optional[str]:
        """Return the live id of ``kind`` for ``symbol`` (or ``None``)."""
        sym = str(symbol or "").strip().upper()
        with self._lock:
            return self._by_symbol.get(sym, {}).get(kind)

    def tracked_ids(self) -> List[str]:
        with self._lock:
            return list(self._orders)
