"""Persistent trade ledger used by the end-of-day report.

Alpaca does not expose realised P/L per filled order, so the bot keeps its own
small FIFO ledger:

* every **buy** fill opens a lot (quantity + fill price),
* every **sell** fill consumes lots for that symbol and records the realised
  profit/loss of the matched shares.

The ledger is written atomically to ``TRADE_LEDGER_FILE`` so a restart during
the day does not lose the trade outcomes. Open lots are *not* cleared at
midnight: a position bought yesterday and sold today still produces a correct
result for today's report. Closed trades are pruned after ``keep_days``.

The module is dependency-free (standard library only) and thread-safe; fills
arrive from the broker callback thread while the main loop may read stats.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Floating point tolerances for quantities and P/L comparisons.
_QTY_EPS = 1e-9
_PL_EPS = 1e-9

# How many days of closed trades to keep in the JSON file.
DEFAULT_KEEP_DAYS = 90


def _enum_value(value: Any) -> str:
    """Normalise ``OrderSide.BUY`` / ``"buy"`` / ``None`` to a bare string."""
    value = getattr(value, "value", value)
    return "" if value is None else str(value).strip().lower()


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _fill_sort_key(fill: Dict[str, Any]) -> float:
    """Chronological sort key for broker fills (missing dates sort first)."""
    moment = _parse_datetime(fill.get("filled_at"))
    if moment is None:
        return 0.0
    try:
        return moment.timestamp()
    except (OverflowError, OSError, ValueError):
        return 0.0


class DailyTradeLedger:
    """FIFO ledger of open lots and realised sell trades.

    Parameters
    ----------
    path:
        JSON file used for persistence. ``None`` keeps the ledger in memory
        (useful for tests).
    keep_days:
        Number of days of closed-trade history retained in the file.
    clock:
        Optional callable returning a timezone-aware ``datetime``; injectable
        for deterministic tests.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        keep_days: int = DEFAULT_KEEP_DAYS,
        clock=None,
    ) -> None:
        self.path = path
        self._keep_days = max(1, int(keep_days))
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._lock = threading.RLock()
        # symbol -> list of {"qty", "price", "opened_at", "seeded"}
        self._lots: Dict[str, List[Dict[str, Any]]] = {}
        # Chronological list of realised (or partially realised) sells.
        self._closed: List[Dict[str, Any]] = []
        # Broker order id -> trade date (YYYY-MM-DD), used to make the
        # intraday callback and the end-of-day reconciliation idempotent.
        self._processed: Dict[str, str] = {}
        # symbol -> ISO timestamp when its opening position was seeded.
        self._seed_times: Dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _now(self) -> datetime:
        return self._clock()

    @staticmethod
    def _day_of(moment: datetime) -> str:
        return moment.date().isoformat()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # corrupt file / permissions
            logger.warning("Could not read trade ledger %s: %s", self.path, exc)
            return

        lots = payload.get("lots")
        closed = payload.get("closed")
        if isinstance(lots, dict):
            for symbol, symbol_lots in lots.items():
                if not isinstance(symbol_lots, list):
                    continue
                cleaned = []
                for lot in symbol_lots:
                    qty = _as_float(lot.get("qty"))
                    price = _as_float(lot.get("price"))
                    if qty is None or price is None or qty <= _QTY_EPS:
                        continue
                    cleaned.append(
                        {
                            "qty": qty,
                            "price": price,
                            "opened_at": lot.get("opened_at"),
                            "seeded": bool(lot.get("seeded", False)),
                        }
                    )
                if cleaned:
                    self._lots[str(symbol).upper()] = cleaned
        if isinstance(closed, list):
            self._closed = [trade for trade in closed if isinstance(trade, dict)]

        processed = payload.get("processed")
        if isinstance(processed, dict):
            self._processed = {
                str(key): str(value)
                for key, value in processed.items()
                if key
            }
        elif isinstance(processed, list):
            # Backwards compatibility with an earlier list-only format.
            self._processed = {str(key): "" for key in processed if key}

        seed_times = payload.get("seed_times")
        if isinstance(seed_times, dict):
            self._seed_times = {
                str(key): str(value) for key, value in seed_times.items() if key
            }

    def _save(self) -> None:
        if not self.path:
            return
        payload = {
            "version": 1,
            "lots": self._lots,
            "closed": self._closed,
            "processed": self._processed,
            "seed_times": self._seed_times,
        }
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".trade_ledger.", suffix=".tmp", dir=directory
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self.path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as exc:  # never break trading because of persistence
            logger.warning("Could not save trade ledger %s: %s", self.path, exc)

    def _prune_locked(self, now: datetime) -> None:
        cutoff = now.timestamp() - self._keep_days * 86400
        kept = []
        for trade in self._closed:
            closed_at = _parse_datetime(trade.get("closed_at"))
            if closed_at is None or closed_at.timestamp() >= cutoff:
                kept.append(trade)
        self._closed = kept

        # Processed-order ids and seed timestamps are only needed for a short
        # window; drop entries older than the retention period.
        cutoff_day = (now - timedelta(days=self._keep_days)).date().isoformat()
        self._processed = {
            key: day
            for key, day in self._processed.items()
            if not day or day >= cutoff_day
        }
        self._seed_times = {
            symbol: stamp
            for symbol, stamp in self._seed_times.items()
            if not stamp or stamp[:10] >= cutoff_day
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_fill(
        self,
        symbol: Any,
        side: Any,
        quantity: Any,
        price: Any,
        when: Optional[datetime] = None,
        fill_id: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Record one filled order and return a summary of the operation.

        Buys append a lot. Sells consume FIFO lots and return the realised
        P/L. If a sell exceeds the known lots (e.g. a position opened before
        the ledger existed), the unmatched quantity is reported as
        ``unknown_qty`` instead of fabricating a result.

        ``fill_id`` is the broker order id. Supplying it makes the method
        idempotent, so the real-time callback and the end-of-day reconciliation
        can safely process the same fill twice.
        """
        symbol = str(symbol or "").strip().upper()
        side = _enum_value(side)
        qty = _as_float(quantity)
        fill_price = _as_float(price)
        if not symbol or side not in {"buy", "sell"}:
            return None
        if qty is None or fill_price is None or qty <= _QTY_EPS or fill_price <= 0:
            return None

        moment = when or self._now()
        order_key = str(fill_id).strip() if fill_id else None
        with self._lock:
            if order_key and order_key in self._processed:
                return None

            if side == "buy":
                self._lots.setdefault(symbol, []).append(
                    {
                        "qty": qty,
                        "price": fill_price,
                        "opened_at": moment.isoformat(),
                        "seeded": False,
                    }
                )
                if order_key:
                    self._processed[order_key] = self._day_of(moment)
                self._save()
                return {
                    "side": "buy",
                    "symbol": symbol,
                    "qty": qty,
                    "price": fill_price,
                    "date": self._day_of(moment),
                }

            remaining = qty
            matched: List[tuple] = []
            lots = self._lots.get(symbol, [])
            while remaining > _QTY_EPS and lots:
                lot = lots[0]
                take = min(remaining, lot["qty"])
                matched.append((take, lot["price"]))
                lot["qty"] -= take
                remaining -= take
                if lot["qty"] <= _QTY_EPS:
                    lots.pop(0)
            if not lots:
                self._lots.pop(symbol, None)

            matched_qty = qty - remaining
            realised = (
                sum((fill_price - entry) * part for part, entry in matched)
                if matched_qty > _QTY_EPS
                else 0.0
            )
            trade = {
                "symbol": symbol,
                "side": "sell",
                "qty": qty,
                "matched_qty": matched_qty,
                "unknown_qty": max(0.0, remaining),
                "entry_price": (
                    sum(part * entry for part, entry in matched) / matched_qty
                    if matched_qty > _QTY_EPS
                    else None
                ),
                "exit_price": fill_price,
                "realized_pl": realised if matched_qty > _QTY_EPS else None,
                "date": self._day_of(moment),
                "closed_at": moment.isoformat(),
            }
            self._closed.append(trade)
            if order_key:
                self._processed[order_key] = self._day_of(moment)
            self._prune_locked(moment)
            self._save()
            return trade

    def seed_positions(
        self,
        positions: Iterable[Dict[str, Any]],
        when: Optional[datetime] = None,
    ) -> List[str]:
        """Seed lots for positions the ledger has never seen.

        Called once per trading day. It only touches symbols with no tracked
        open quantity, so it never overwrites fills recorded by the bot.
        Returns the list of symbols that were seeded.
        """
        moment = when or self._now()
        seeded: List[str] = []
        with self._lock:
            for position in positions or []:
                symbol = str(position.get("symbol") or "").strip().upper()
                qty = _as_float(position.get("qty"))
                entry = _as_float(position.get("avg_entry"))
                if not symbol or qty is None or entry is None:
                    continue
                if qty <= _QTY_EPS or entry <= 0:
                    continue
                tracked = sum(lot["qty"] for lot in self._lots.get(symbol, []))
                if tracked > _QTY_EPS:
                    continue
                self._lots[symbol] = [
                    {
                        "qty": qty,
                        "price": entry,
                        "opened_at": moment.isoformat(),
                        "seeded": True,
                    }
                ]
                self._seed_times[symbol] = moment.isoformat()
                seeded.append(symbol)
            if seeded:
                self._save()
        return seeded

    def _is_before_seed(self, symbol: str, when: Optional[datetime]) -> bool:
        """True when a fill predates the seeded snapshot for ``symbol``.

        This prevents double counting when the bot starts mid-session: the
        current position is seeded with its average entry, while Alpaca still
        reports the buys that created it earlier in the day.
        """
        if when is None:
            return False
        with self._lock:
            stamp = self._seed_times.get(symbol)
        seed_at = _parse_datetime(stamp)
        if seed_at is None:
            return False
        try:
            return when < seed_at
        except TypeError:
            # One value is naive and the other aware; compare wall-clock time.
            try:
                return when.replace(tzinfo=None) < seed_at.replace(tzinfo=None)
            except Exception:
                return False

    def reconcile_fills(self, fills: Iterable[Dict[str, Any]]) -> int:
        """Import broker fills in chronological order, skipping duplicates.

        Used at market close so bracket exit legs (which lumibot does not
        report through ``on_filled_order``) are included in the daily win/loss
        counts. Returns the number of newly recorded fills.
        """
        ordered = sorted(
            (fill for fill in fills or [] if isinstance(fill, dict)),
            key=_fill_sort_key,
        )
        recorded = 0
        for fill in ordered:
            symbol = str(fill.get("symbol") or "").strip().upper()
            when = _parse_datetime(fill.get("filled_at"))
            if symbol and self._is_before_seed(symbol, when):
                continue
            trade = self.record_fill(
                symbol,
                fill.get("side"),
                fill.get("qty"),
                fill.get("price"),
                when=when,
                fill_id=fill.get("id"),
            )
            if trade is not None:
                recorded += 1
        return recorded

    def stats(self, day: Optional[str] = None) -> Dict[str, Any]:
        """Return win/loss counts and realised P/L for ``day`` (``YYYY-MM-DD``).

        ``unknown`` counts sells that could not be matched to an entry (for
        example a position opened before the ledger file existed). Trades with
        a partially matched quantity are still classified by the P/L of the
        matched shares.
        """
        day = day or self._day_of(self._now())
        with self._lock:
            trades = [
                dict(trade)
                for trade in self._closed
                if str(trade.get("date")) == day
            ]

        wins = losses = breakeven = unknown = 0
        realised_total = 0.0
        for trade in trades:
            realised = _as_float(trade.get("realized_pl"))
            if realised is None:
                unknown += 1
                continue
            realised_total += realised
            if realised > _PL_EPS:
                wins += 1
            elif realised < -_PL_EPS:
                losses += 1
            else:
                breakeven += 1

        return {
            "date": day,
            "trade_count": len(trades),
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "unknown": unknown,
            "realized_pl": realised_total,
            "trades": trades,
        }
