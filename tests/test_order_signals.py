"""Tests for 5-minute signal confirmation, the hold-time guard and order state.

Run from the repository root with ``python -m pytest``.
"""

import logging
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tradingbot as t  # noqa: E402  (path setup must run first)
from lumibot.trading_builtins import SafeList  # noqa: E402
from order_state import STOP_LOSS, TAKE_PROFIT, OrderTracker  # noqa: E402


def _signal(**overrides):
    data = {
        "symbol": "AAPL",
        "price": 100.0,
        "rsi": t.RSI_OVERSOLD,
        "trend_bull": True,
        "macd_bull": True,
        "sentiment_label": "positive",
        "sentiment_score": t.MIN_SENTIMENT_CONFIDENCE,
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Strict entry confirmation
# ---------------------------------------------------------------------------
def test_entry_requires_all_legs_to_align():
    assert t._is_entry_signal(_signal()) is True


@pytest.mark.parametrize(
    "override",
    [
        {"trend_bull": False},
        {"macd_bull": False},
        {"rsi": t.RSI_OVERSOLD + 1.0},
        {"sentiment_label": "negative"},
        {"sentiment_label": "neutral"},
        {"sentiment_score": t.MIN_SENTIMENT_CONFIDENCE - 0.01},
    ],
)
def test_entry_is_rejected_when_one_leg_disagrees(override):
    assert t._is_entry_signal(_signal(**override)) is False


def test_entry_requires_rsi_and_sentiment_values():
    data = _signal()
    data.pop("rsi")
    assert t._is_entry_signal(data) is False

    data = _signal()
    data.pop("sentiment_score")
    assert t._is_entry_signal(data) is False


# ---------------------------------------------------------------------------
# Order tracker
# ---------------------------------------------------------------------------
def test_order_tracker_replaces_and_forgets_ids():
    tracker = OrderTracker()
    tracker.register("stop-1", "aapl", STOP_LOSS, "new")
    tracker.register("tp-1", "AAPL", TAKE_PROFIT, "new")

    assert tracker.order_id("AAPL", STOP_LOSS) == "stop-1"
    assert tracker.order_id("AAPL", TAKE_PROFIT) == "tp-1"
    assert tracker.info("stop-1")["symbol"] == "AAPL"

    # A replaced stop keeps only the new id.
    tracker.register("stop-2", "AAPL", STOP_LOSS, "accepted")
    assert tracker.order_id("AAPL", STOP_LOSS) == "stop-2"
    assert tracker.info("stop-1") is None

    tracker.forget("stop-2")
    assert tracker.order_id("AAPL", STOP_LOSS) is None
    assert tracker.order_id("AAPL", TAKE_PROFIT) == "tp-1"


def test_order_tracker_sync_purges_closed_legs():
    tracker = OrderTracker()
    tracker.register("stop-1", "AAPL", STOP_LOSS, "new")
    tracker.register("tp-1", "AAPL", TAKE_PROFIT, "new")
    tracker.register("stop-9", "MSFT", STOP_LOSS, "new")

    stale = tracker.sync(
        [
            {"id": "stop-1", "symbol": "AAPL", "kind": STOP_LOSS, "status": "new"},
        ]
    )

    assert sorted(stale) == ["stop-9", "tp-1"]
    assert tracker.tracked_ids() == ["stop-1"]
    assert tracker.info("tp-1") is None
    assert tracker.order_id("MSFT", STOP_LOSS) is None


def test_order_tracker_ignores_unknown_legs_and_empty_ids():
    tracker = OrderTracker()
    tracker.register("entry-1", "AAPL", "entry", "new")
    tracker.register("", "AAPL", STOP_LOSS, "new")
    tracker.register("stop-1", "", STOP_LOSS, "new")

    assert tracker.tracked_ids() == []
    tracker.sync([{"id": "entry-1", "symbol": "AAPL", "kind": "entry"}])
    assert tracker.tracked_ids() == []


def test_order_tracker_forget_symbol_clears_all_legs():
    tracker = OrderTracker()
    tracker.register("stop-1", "AAPL", STOP_LOSS)
    tracker.register("tp-1", "AAPL", TAKE_PROFIT)
    tracker.register("stop-2", "MSFT", STOP_LOSS)

    tracker.forget_symbol("aapl")

    assert tracker.tracked_ids() == ["stop-2"]
    assert tracker.order_id("AAPL", STOP_LOSS) is None
    assert tracker.order_id("AAPL", TAKE_PROFIT) is None


# ---------------------------------------------------------------------------
# lumibot stale-order purge
# ---------------------------------------------------------------------------
def _safe_list(*orders):
    return SafeList(threading.RLock(), list(orders))


def _order(identifier, side="sell", order_type="stop", status="new"):
    return SimpleNamespace(
        identifier=identifier,
        symbol="AAPL",
        side=side,
        type=order_type,
        status=status,
    )


def test_closed_orders_are_purged_from_lumibot_tracking():
    bot = object.__new__(t.MLTraderProMax)
    live = _order("live-1")
    stale = _order("stale-1")
    filled = _order("filled-1", side="buy", order_type="market", status="filled")
    bot.broker = SimpleNamespace(
        _unprocessed_orders=_safe_list(stale, live),
        _new_orders=_safe_list(filled),
        _partially_filled_orders=_safe_list(),
    )

    removed = bot._purge_stale_broker_orders([live])

    assert removed == 2
    assert [order.identifier for order in bot.broker._unprocessed_orders] == ["live-1"]
    assert list(bot.broker._new_orders) == []


def test_replaced_order_id_is_forgotten_immediately():
    bot = object.__new__(t.MLTraderProMax)
    old = _order("old-stop")
    new = _order("new-stop")
    bot.broker = SimpleNamespace(
        _unprocessed_orders=_safe_list(old, new),
        _new_orders=_safe_list(),
        _partially_filled_orders=_safe_list(),
    )

    bot._forget_broker_order_id("old-stop")

    assert [order.identifier for order in bot.broker._unprocessed_orders] == [
        "new-stop"
    ]


# ---------------------------------------------------------------------------
# Minimum hold time
# ---------------------------------------------------------------------------
def _exit_bot(now, opened_at):
    bot = object.__new__(t.MLTraderProMax)
    bot._log = logging.getLogger("hold-test")
    bot._position_opened_at = {"AAPL": opened_at.isoformat()} if opened_at else {}
    bot._market_now = lambda: now
    bot._closed = []

    def _close(symbol, quantity, reason, price=None):
        bot._closed.append((symbol, quantity, price))
        return True

    bot._close_position_market = _close
    return bot


def test_hold_time_blocks_a_signal_exit_until_fifteen_minutes():
    now = datetime(2026, 10, 20, 11, 0, tzinfo=t._market_timezone())
    bot = _exit_bot(now, now - timedelta(minutes=5))
    bot.get_position = lambda symbol: SimpleNamespace(quantity=10)

    bot._try_exit({"symbol": "AAPL", "price": 100.0, "rsi": 80.0})

    assert bot._closed == []
    assert bot._hold_time_elapsed("AAPL", now) is False


def test_signal_exit_runs_after_the_minimum_hold():
    now = datetime(2026, 10, 20, 11, 0, tzinfo=t._market_timezone())
    bot = _exit_bot(now, now - timedelta(minutes=16))
    bot.get_position = lambda symbol: SimpleNamespace(quantity=10)

    bot._try_exit({"symbol": "AAPL", "price": 100.0, "rsi": 80.0})

    assert bot._closed == [("AAPL", 10, 100.0)]


def test_unknown_open_time_does_not_trap_a_position():
    now = datetime(2026, 10, 20, 11, 0, tzinfo=t._market_timezone())
    bot = _exit_bot(now, None)

    assert bot._hold_time_elapsed("AAPL", now) is True


# ---------------------------------------------------------------------------
# Cancellation alerts
# ---------------------------------------------------------------------------
def _cancel_bot():
    bot = object.__new__(t.MLTraderProMax)
    bot._log = logging.getLogger("cancel-test")
    bot._order_tracker = OrderTracker()
    bot._auto_cancels = {}
    bot._alerts = []
    bot.telegram_send_message = (
        lambda message, wait=False: bot._alerts.append(message)
    )
    bot._forget_broker_order = lambda order: None
    return bot


def test_protective_leg_cancel_stays_silent():
    bot = _cancel_bot()
    order = SimpleNamespace(
        identifier="stop-1", symbol="AAPL", side="sell", type="stop", status="canceled"
    )
    bot._order_tracker.register("stop-1", "AAPL", STOP_LOSS)

    bot.on_canceled_order(order)

    assert bot._alerts == []
    assert bot._order_tracker.info("stop-1") is None


def test_bot_requested_cancel_stays_silent():
    bot = _cancel_bot()
    bot._auto_cancels["buy-1"] = 1.0
    order = SimpleNamespace(
        identifier="buy-1", symbol="AAPL", side="buy", type="market", status="canceled"
    )

    bot.on_canceled_order(order)

    assert bot._alerts == []
    assert "buy-1" not in bot._auto_cancels


def test_genuine_cancel_is_reported_to_telegram():
    bot = _cancel_bot()
    order = SimpleNamespace(
        identifier="buy-2", symbol="AAPL", side="buy", type="limit", status="canceled"
    )

    bot.on_canceled_order(order)

    assert len(bot._alerts) == 1
    assert "تم إلغاء الأمر" in bot._alerts[0]
