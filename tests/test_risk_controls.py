"""Tests for the strict risk controls added after the drawdown.

Run from the repository root with ``python -m pytest``.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tradingbot as t  # noqa: E402  (path setup must run first)


# ---------------------------------------------------------------------------
# Stop loss band
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("price", [2.0, 5.0, 17.43, 100.0, 1234.56])
def test_bracket_stop_is_always_inside_the_hard_band(monkeypatch, price):
    for configured in (0.90, 0.98, 0.985, 0.999):
        monkeypatch.setattr(t, "STOP_LOSS_PCT", configured)
        stop, take_profit = t.bracket_prices(price)
        distance = (price - stop) / price
        assert 0.010 - 1e-9 <= distance <= 0.015 + 1e-9
        assert stop < price < take_profit


def test_penny_stock_without_a_valid_stop_is_rejected():
    with pytest.raises(ValueError):
        t.bracket_prices(0.35)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
def test_position_size_caps_notional_at_2500():
    stop, _ = t.bracket_prices(100.0)
    qty = t.position_size(100.0, cash=1_000_000, equity=1_000_000, stop_loss_price=stop)
    assert qty == 25  # $2,500 / $100


def test_position_size_caps_notional_at_3pct_of_cash():
    stop, _ = t.bracket_prices(100.0)
    qty = t.position_size(100.0, cash=50_000, equity=50_000, stop_loss_price=stop)
    assert qty == 15  # 3% of $50,000 = $1,500


# ---------------------------------------------------------------------------
# Break-even / trailing stop
# ---------------------------------------------------------------------------
def test_break_even_moves_stop_above_entry():
    level = t.trailing_stop_level(entry=100.0, current=100.6)
    assert level >= 100.10


def test_trailing_stop_follows_the_high_water_mark():
    level = t.trailing_stop_level(
        entry=100.0, current=104.0, high_water=105.0
    )
    assert level >= 104.16


def test_trailing_stop_never_moves_down():
    level = t.trailing_stop_level(
        entry=100.0, current=101.2, high_water=101.2, existing_stop=102.5
    )
    assert level >= 102.5


# ---------------------------------------------------------------------------
# Daily loss limit
# ---------------------------------------------------------------------------
def test_daily_limit_uses_the_stricter_threshold():
    assert t.effective_daily_loss_limit(100_000) == pytest.approx(250.0)
    assert t.effective_daily_loss_limit(50_000) == pytest.approx(125.0)
    assert t.effective_daily_loss_limit(1_000_000) == pytest.approx(250.0)


def test_circuit_breaker_message_matches_the_requested_wording():
    message = t.format_circuit_breaker_message(250)
    assert "توقف حماية الحساب (Circuit Breaker)" in message
    assert "$250" in message
    assert "تم إيقاف التداول اليوم" in message


# ---------------------------------------------------------------------------
# Circuit breaker behaviour
# ---------------------------------------------------------------------------
class _FakeLedger:
    def __init__(self, realized=0.0):
        self.realized = realized

    def stats(self, day):
        return {"realized_pl": self.realized}


def _risk_bot(
    monkeypatch,
    equity,
    last_equity,
    realized=0.0,
    unrealized=None,
    position_error=False,
):
    bot = object.__new__(t.MLTraderProMax)
    bot._log = logging.getLogger("risk-test")
    bot._risk_state_path = None
    bot._risk_state = {}
    bot._high_water = {}
    bot._software_stops = {}
    bot._closing_symbols = set()
    bot._risk_data_stale = False
    bot._circuit_breaker_handled = False
    bot._last_loss_warning = None
    bot._trade_ledger = _FakeLedger(realized)
    bot.broker = SimpleNamespace(api=object())
    bot._is_backtesting = False

    positions = []
    if unrealized is not None:
        positions.append(
            {
                "symbol": "AAPL",
                "qty": 10,
                "avg_entry": 100.0,
                "current_price": 90.0,
                "market_value": 900.0,
                "unrealized_pl": unrealized,
                "unrealized_plpc": -0.1,
            }
        )

    def _positions(strict=False):
        if position_error:
            raise RuntimeError("positions endpoint unavailable")
        return positions

    bot._position_rows = _positions
    bot._liquidate_calls = []
    bot._alerts = []

    def _liquidate(reason):
        bot._liquidate_calls.append(reason)
        count = len(positions)
        positions.clear()  # a successful liquidation flattens the account
        return count

    bot._liquidate_all_positions = _liquidate
    bot.telegram_send_message = (
        lambda message, wait=False: bot._alerts.append(message)
    )

    monkeypatch.setattr(
        t,
        "_account_snapshot",
        lambda api: {
            "status": "active",
            "cash": equity,
            "equity": equity,
            "portfolio_value": equity,
            "buying_power": equity,
            "long_market_value": 0.0,
            "short_market_value": 0.0,
            "last_equity": last_equity,
            "day_pl": equity - last_equity,
            "day_pl_pct": (equity - last_equity) / last_equity * 100.0,
        },
    )
    return bot


def test_circuit_breaker_liquidates_and_alerts(monkeypatch):
    bot = _risk_bot(
        monkeypatch,
        equity=99_700.0,
        last_equity=100_000.0,
        realized=-100.0,
        unrealized=-200.0,
    )
    now = datetime(2026, 10, 20, 11, 0, tzinfo=t._market_timezone())

    assert bot._check_circuit_breaker(now) is True
    assert bot._risk_state["circuit_breaker_tripped"] is True
    assert bot._liquidate_calls == ["daily loss limit"]
    assert len(bot._alerts) == 1
    assert "Circuit Breaker" in bot._alerts[0]

    # Idempotent: a second iteration must not liquidate or alert again.
    assert bot._check_circuit_breaker(now) is True
    assert len(bot._liquidate_calls) == 1
    assert len(bot._alerts) == 1


def test_circuit_breaker_stays_quiet_below_the_limit(monkeypatch):
    bot = _risk_bot(
        monkeypatch,
        equity=99_900.0,
        last_equity=100_000.0,
        realized=-50.0,
        unrealized=-50.0,
    )
    now = datetime(2026, 10, 20, 11, 0, tzinfo=t._market_timezone())

    assert bot._check_circuit_breaker(now) is False
    assert bot._liquidate_calls == []
    assert bot._alerts == []


def test_circuit_breaker_halt_survives_restart(monkeypatch, tmp_path):
    state_file = tmp_path / "risk_state.json"
    bot = _risk_bot(
        monkeypatch,
        equity=99_700.0,
        last_equity=100_000.0,
        realized=-100.0,
        unrealized=-200.0,
    )
    bot._risk_state_path = str(state_file)
    now = datetime(2026, 10, 20, 11, 0, tzinfo=t._market_timezone())

    assert bot._check_circuit_breaker(now) is True
    assert state_file.exists()

    # A fresh process on the same day must remain halted.
    restarted = _risk_bot(
        monkeypatch,
        equity=99_700.0,
        last_equity=100_000.0,
        realized=-100.0,
        unrealized=-200.0,
    )
    restarted._risk_state_path = str(state_file)
    restarted._risk_state = restarted._load_risk_state()
    assert restarted._trading_halted(now) is True

    # The next trading day starts with a clean slate.
    next_day = now + timedelta(days=1)
    assert restarted._trading_halted(next_day) is False


# ---------------------------------------------------------------------------
# Maximum concurrent positions
# ---------------------------------------------------------------------------
def test_exposure_slots_count_positions_and_pending_entries():
    bot = object.__new__(t.MLTraderProMax)
    bot._position_rows = lambda strict=False: [
        {"symbol": "AAPL", "qty": 10},
        {"symbol": "MSFT", "qty": 5},
    ]
    orders = [
        SimpleNamespace(symbol="TSLA", side="buy", status="new"),
        SimpleNamespace(symbol="NVDA", side="buy", status="filled"),
        SimpleNamespace(symbol="AAPL", side="buy", status="new"),
    ]
    bot._open_orders = lambda: orders

    assert bot._count_exposure_slots() == 3

    orders.pop(0)
    assert bot._count_exposure_slots() == 2


def test_exposure_slots_fail_closed_when_positions_cannot_be_read(monkeypatch):
    bot = _risk_bot(
        monkeypatch, equity=100_000.0, last_equity=100_000.0, position_error=True
    )
    assert bot._count_exposure_slots() == t.MAX_CONCURRENT_POSITIONS


# ---------------------------------------------------------------------------
# Fail-closed behaviour and data quality
# ---------------------------------------------------------------------------
def test_daily_pnl_uses_the_broker_day_window(monkeypatch):
    """A position held from yesterday must not count as today's loss."""
    bot = _risk_bot(
        monkeypatch,
        equity=100_000.0,
        last_equity=100_000.0,
        realized=-400.0,
        unrealized=-400.0,
    )
    now = datetime(2026, 10, 20, 11, 0, tzinfo=t._market_timezone())

    snapshot = bot._daily_pnl_snapshot(now)
    assert snapshot["total"] == pytest.approx(0.0)
    assert snapshot["unrealized"] == pytest.approx(-400.0)
    assert snapshot["data_ok"] is True
    assert bot._check_circuit_breaker(now) is False
    assert bot._liquidate_calls == []


def test_unreadable_account_data_blocks_new_entries(monkeypatch):
    bot = _risk_bot(
        monkeypatch, equity=100_000.0, last_equity=100_000.0, position_error=True
    )

    def _broken_account(api):
        raise RuntimeError("account endpoint unavailable")

    monkeypatch.setattr(t, "_account_snapshot", _broken_account)
    now = datetime(2026, 10, 20, 11, 0, tzinfo=t._market_timezone())

    assert bot._check_circuit_breaker(now) is False
    assert bot._risk_data_stale is True

    # Data still stale: no entry may be attempted.
    assert bot._check_circuit_breaker(now) is False
    assert bot._risk_data_stale is True


def test_corrupt_risk_state_file_halts_the_session(monkeypatch, tmp_path):
    state_file = tmp_path / "risk_state.json"
    state_file.write_text("{this is not json", encoding="utf-8")

    bot = _risk_bot(monkeypatch, equity=100_000.0, last_equity=100_000.0)
    bot._risk_state_path = str(state_file)
    bot._risk_state = bot._load_risk_state()

    now = datetime(2026, 10, 20, 11, 0, tzinfo=t._market_timezone())
    assert bot._risk_state.get("_corrupt") is True
    assert bot._trading_halted(now) is True
    # The corrupt file is replaced with a valid, tripped state.
    assert bot._load_risk_state().get("circuit_breaker_tripped") is True


def test_position_read_failure_keeps_software_stops_armed(monkeypatch):
    bot = object.__new__(t.MLTraderProMax)
    bot._log = logging.getLogger("risk-test")
    bot._software_stops = {"AAPL": 101.0}
    bot._high_water = {"AAPL": 102.0}
    bot._closing_symbols = set()

    def _broken(strict=False):
        raise RuntimeError("positions endpoint unavailable")

    bot._position_rows = _broken
    now = datetime(2026, 10, 20, 11, 0, tzinfo=t._market_timezone())

    bot._manage_open_positions(now)
    assert bot._software_stops == {"AAPL": 101.0}
    assert bot._high_water == {"AAPL": 102.0}


def test_unacknowledged_order_is_treated_as_a_rejection():
    assert (
        t.MLTraderProMax._order_failure(
            SimpleNamespace(status="unprocessed", _error=None)
        )
        == "unprocessed"
    )
    assert (
        t.MLTraderProMax._order_failure(SimpleNamespace(status="new", _error=None))
        is None
    )
    assert (
        t.MLTraderProMax._order_failure(
            SimpleNamespace(status="error", _error="insufficient buying power")
        )
        == "insufficient buying power"
    )


def test_working_market_sell_blocks_a_duplicate_close():
    bot = object.__new__(t.MLTraderProMax)
    bot.broker = SimpleNamespace(api=None)
    bot._open_orders = lambda: [
        SimpleNamespace(symbol="AAPL", side="sell", status="new", type="stop")
    ]
    # A protective stop must not block an intentional market exit.
    assert bot._has_open_sell_orders("AAPL", market_only=True) is False

    bot._open_orders = lambda: [
        SimpleNamespace(symbol="AAPL", side="sell", status="new", type="market")
    ]
    assert bot._has_open_sell_orders("AAPL", market_only=True) is True


def test_trailing_stop_stays_armed_after_a_pullback():
    # The position peaked at +1.5%, so the trail remains armed even though the
    # current price has fallen back to +0.2%.
    level = t.trailing_stop_level(entry=100.0, current=100.2, high_water=101.5)
    assert level >= 100.1
    assert level > 100.2  # the caller must close at market
