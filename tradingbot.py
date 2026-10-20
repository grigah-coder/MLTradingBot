"""Live/paper ML trading bot (production-grade).

Design goals
------------
* **Structured logging** - no ``print``; a rotating file + console logger with
  INFO/WARNING/ERROR levels (see :mod:`logging_setup`).
* **Resilient networking** - every outbound call (Alpaca news, historical
  prices, order submission, Telegram) is wrapped with exponential backoff and
  jitter, and understands Alpaca rate-limit/5xx errors (see :mod:`retry_utils`).
* **Concurrency** - symbols in ``WATCHLIST`` are analysed in parallel; order
  placement stays on the main thread to keep broker state consistent.
* **Caching** - headlines, sentiment and indicators are memoised with a TTL.
* **Fault isolation** - a failure on one symbol is logged and skipped; it can
  never halt the main loop.
"""

from __future__ import annotations

import atexit
import hashlib
import html
import json
import logging
import math
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import requests # pyright: ignore[reportMissingModuleSource]
from alpaca.data.historical.news import NewsClient # pyright: ignore[reportMissingImports]
from alpaca.data.requests import NewsRequest # pyright: ignore[reportMissingImports]
from lumibot.brokers import Alpaca # pyright: ignore[reportMissingImports]

# Trading-side alpaca-py imports (used by the end-of-day summary). Kept soft so
# a leaner alpaca-py install cannot break importing the whole bot.
try:  # pragma: no cover - depends on alpaca-py version
    from alpaca.trading.enums import QueryOrderStatus # pyright: ignore[reportMissingImports]
    from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    QueryOrderStatus = None  # type: ignore
    GetOrdersRequest = None  # type: ignore
    ReplaceOrderRequest = None  # type: ignore
from lumibot.strategies.strategy import Strategy # pyright: ignore[reportMissingImports]
from lumibot.traders import Trader # pyright: ignore[reportMissingImports]

from alpaca_data_source import AlpacaIEXData
from cache_utils import TTLCache
from finbert_utils import estimate_sentiment, warm_up as finbert_warm_up
from logging_setup import configure_logging
from retry_utils import call_with_retry
from trade_ledger import DailyTradeLedger
from config import (
    WATCHLIST,
    NEWS_CACHE_TTL,
    SENTIMENT_CACHE_TTL,
    INDICATOR_CACHE_TTL,
    CACHE_MAXSIZE,
    MAX_WORKERS,
    ALERT_WORKERS,
    INDICATOR_LOOKBACK,
    MIN_BARS_REQUIRED,
    NEWS_LIMIT,
    RSI_PERIOD,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    POSITION_SIZE_PCT,
    MAX_POSITION_NOTIONAL,
    MAX_PORTFOLIO_RISK_PCT,
    MAX_SHARES_PER_TRADE,
    MAX_CONCURRENT_POSITIONS,
    STOP_LOSS_MIN_PCT,
    STOP_LOSS_MAX_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MIN_REWARD_RISK_RATIO,
    TRAILING_STOP_ENABLED,
    BREAKEVEN_TRIGGER_PCT,
    BREAKEVEN_BUFFER_PCT,
    TRAILING_TRIGGER_PCT,
    TRAILING_STOP_PCT,
    DAILY_LOSS_LIMIT_ENABLED,
    DAILY_LOSS_LIMIT_USD,
    DAILY_LOSS_LIMIT_PCT,
    RISK_STATE_FILE,
    TIME_FILTER_ENABLED,
    MARKET_OPEN_TIME,
    OPENING_BLACKOUT_MINUTES,
    ORDER_TYPE,
    REQUIRE_PROTECTIVE_ORDERS,
    BRACKET_FALLBACK_TO_MARKET,
    ORDER_REGISTRATION_TIMEOUT,
    TELEGRAM_ENABLED,
    TELEGRAM_TIMEOUT,
    TELEGRAM_RETRY_ATTEMPTS,
    SESSION_START_MESSAGE_ENABLED,
    DAILY_SUMMARY_ENABLED,
    MARKET_TIMEZONE,
    TRADE_LEDGER_FILE,
    RETRY_ATTEMPTS,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    RETRY_BACKOFF_FACTOR,
    RETRY_JITTER,
    SLEEPTIME,
    ALPACA_DATA_FEED,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
    IS_BACKTESTING,
    BACKTEST_START,
    BACKTEST_END,
    alpaca_config,
)

# Logging is configured early and idempotently so worker threads inherit it.
configure_logging()
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Small helper so every retry uses the same configured policy.
# --------------------------------------------------------------------------
def _retry(func, *args, description=None, attempts=None, **kwargs):
    return call_with_retry(
        func,
        *args,
        attempts=attempts or RETRY_ATTEMPTS,
        base_delay=RETRY_BASE_DELAY,
        max_delay=RETRY_MAX_DELAY,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        jitter=RETRY_JITTER,
        description=description,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Module-level, process-wide caches (shared safely across worker threads).
# --------------------------------------------------------------------------
_news_cache = TTLCache(ttl=NEWS_CACHE_TTL, maxsize=CACHE_MAXSIZE)
_sentiment_cache = TTLCache(ttl=SENTIMENT_CACHE_TTL, maxsize=CACHE_MAXSIZE)
_indicator_cache = TTLCache(ttl=INDICATOR_CACHE_TTL, maxsize=CACHE_MAXSIZE)

# Reuse a single NewsClient instead of constructing one per symbol/call.
_news_client = None
_news_client_lock = threading.Lock()

# Fire-and-forget alert pool so Telegram never blocks the trading loop.
_alert_pool = ThreadPoolExecutor(max_workers=ALERT_WORKERS, thread_name_prefix="alert")
atexit.register(_alert_pool.shutdown, wait=False)


def _get_news_client() -> NewsClient:
    """Return the shared NewsClient, creating it lazily and thread-safely."""
    global _news_client
    if _news_client is None:
        with _news_client_lock:
            if _news_client is None:
                _news_client = NewsClient(
                    api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET
                )
                logger.info("Alpaca NewsClient initialised")
    return _news_client


def _digest(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()


def _build_news_request(symbol: str, limit: int):
    """Build a NewsRequest across alpaca-py versions.

    Modern alpaca-py uses the ``symbols`` field; ``symbol_or_symbols`` is a
    legacy alias that some versions ignore silently.
    """
    try:
        return NewsRequest(symbols=symbol, limit=limit, sort="desc")
    except TypeError:  # pragma: no cover - legacy alpaca-py only
        return NewsRequest(symbol_or_symbols=symbol, limit=limit, sort="desc")


def _extract_headlines(news, symbol: str) -> str:
    """Extract and join headlines from a NewsSet or a raw dict response."""
    items = getattr(news, "news", None)
    if items is None and isinstance(news, dict):
        items = news.get("news")
    if not items:
        return ""

    headlines = []
    for item in items:
        if isinstance(item, dict):
            headline = item.get("headline")
        else:
            headline = getattr(item, "headline", None)
        if headline:
            headlines.append(headline)
    return " ".join(headlines)


def fetch_headlines(symbol: str, limit: int = NEWS_LIMIT) -> str:
    """Fetch recent headlines for ``symbol``, cached for ``NEWS_CACHE_TTL``.

    Retries transient Alpaca/network failures with backoff. Never raises: on
    final failure a sentinel string is returned (and *not* cached, so the next
    iteration retries).
    """
    cached = _news_cache.get(symbol)
    if cached is not None:
        return cached

    try:
        client = _get_news_client()
        request = _build_news_request(symbol, limit)
        news = _retry(
            client.get_news,
            request,
            description=f"alpaca_news[{symbol}]",
        )
        headlines = _extract_headlines(news, symbol)
        if not headlines:
            headlines = f"No recent news for {symbol}."
    except Exception as exc:
        logger.error("Error fetching news for %s: %s", symbol, exc)
        return f"Error fetching news for {symbol}."

    _news_cache.set(symbol, headlines)
    return headlines


def sentiment_for(symbol: str, headlines: str):
    """Return ``(score, label)`` for a symbol, cached for ``SENTIMENT_CACHE_TTL``.

    The cached value is invalidated automatically when the headline text
    changes. Never raises.
    """
    digest = _digest(headlines)
    cached = _sentiment_cache.get(symbol)
    if cached is not None and cached[0] == digest:
        return cached[1]

    try:
        result = estimate_sentiment(headlines)
    except Exception as exc:
        logger.error("Sentiment error for %s: %s", symbol, exc)
        result = (0.0, "neutral")

    _sentiment_cache.set(symbol, (digest, result))
    return result


def compute_indicators(df: pd.DataFrame):
    """Compute RSI + MACD on a price DataFrame (pure / thread-safe).

    Returns ``None`` when there is not enough data.
    """
    if df is None or df.empty or len(df) < MIN_BARS_REQUIRED:
        return None

    close = df["close"].astype("float64", copy=False)

    # --- RSI -------------------------------------------------------------
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD, min_periods=RSI_PERIOD).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # --- MACD ------------------------------------------------------------
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()

    last_rsi = rsi.iloc[-1]
    if pd.isna(last_rsi):
        return None

    return {
        "rsi": float(last_rsi),
        "macd_bull": bool(macd.iloc[-1] > signal.iloc[-1]),
        "close": float(close.iloc[-1]),
    }


# --------------------------------------------------------------------------
# End-of-day reporting helpers
# --------------------------------------------------------------------------
def _market_timezone():
    """Resolve ``MARKET_TIMEZONE``, falling back to UTC if tzdata is absent."""
    try:
        return ZoneInfo(MARKET_TIMEZONE)
    except Exception:
        logger.warning(
            "Unknown MARKET_TIMEZONE=%r; falling back to UTC", MARKET_TIMEZONE
        )
        return timezone.utc


def _start_of_trading_day(now=None):
    """Return market-local midnight of the current day (tz-aware)."""
    tz = _market_timezone()
    moment = now.astimezone(tz) if now is not None else datetime.now(tz)
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _parse_market_open(value=MARKET_OPEN_TIME):
    """Parse ``MARKET_OPEN_TIME`` (``HH:MM``) into ``(hour, minute)``."""
    try:
        parsed = datetime.strptime(str(value).strip(), "%H:%M")
        return parsed.hour, parsed.minute
    except (TypeError, ValueError):
        logger.warning("Invalid MARKET_OPEN_TIME=%r; falling back to 09:30", value)
        return 9, 30


def opening_blackout_window(now=None):
    """Return ``(start, end)`` of today's opening no-trade window.

    Both datetimes are market-local and tz-aware. Weekends still return a
    window; :func:`in_opening_blackout` is what decides whether it applies.
    """
    tz = _market_timezone()
    moment = now.astimezone(tz) if now is not None else datetime.now(tz)
    hour, minute = _parse_market_open()
    start = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    end = start + timedelta(minutes=max(0, OPENING_BLACKOUT_MINUTES))
    return start, end


def in_opening_blackout(now=None):
    """Return ``True`` while new orders must be suppressed after the open.

    The comparison is done in ``MARKET_TIMEZONE`` (default America/New_York),
    so the window is 09:30-10:00 local time in both EST and EDT. ``now`` must
    be tz-aware when supplied; ``None`` means "right now".
    """
    if not TIME_FILTER_ENABLED or OPENING_BLACKOUT_MINUTES <= 0:
        return False

    tz = _market_timezone()
    moment = now.astimezone(tz) if now is not None else datetime.now(tz)
    if moment.weekday() >= 5:  # Saturday/Sunday: no cash session to protect
        return False

    start, end = opening_blackout_window(moment)
    return start <= moment < end


def _as_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _enum_value(value):
    """Normalise alpaca-py enums (``OrderSide.BUY`` -> ``"buy"``) to a string."""
    value = getattr(value, "value", value)
    return "" if value is None else str(value)


def _money(value):
    return "n/a" if value is None else f"${value:,.2f}"


def _signed_money(value):
    if value is None:
        return "n/a"
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def _truncate(text, limit):
    """Truncate on a line boundary.

    Telegram messages use HTML markup, so cutting mid-tag would make the whole
    message invalid. Keeping complete lines guarantees balanced tags.
    """
    text = str(text)
    if len(text) <= limit:
        return text
    suffix = "\n…"
    kept = []
    used = 0
    for line in text.splitlines():
        if used + len(line) + 1 + len(suffix) > limit:
            break
        kept.append(line)
        used += len(line) + 1
    if not kept:
        # A single oversized line: best effort, without exceeding the limit.
        return text[: max(0, limit - len(suffix))] + suffix
    return "\n".join(kept) + suffix


def _account_snapshot(api):
    """Fetch the Alpaca account balances as a plain dict."""
    account = _retry(api.get_account, description="alpaca_get_account")
    equity = _as_float(getattr(account, "equity", None))
    last_equity = _as_float(getattr(account, "last_equity", None))
    day_pl = (equity - last_equity) if last_equity else None
    day_pl_pct = (
        (day_pl / last_equity * 100.0)
        if day_pl is not None and last_equity
        else None
    )
    return {
        "status": _enum_value(getattr(account, "status", None)),
        "cash": _as_float(getattr(account, "cash", None)),
        "equity": equity,
        "portfolio_value": _as_float(getattr(account, "portfolio_value", None)),
        "buying_power": _as_float(getattr(account, "buying_power", None)),
        "long_market_value": _as_float(getattr(account, "long_market_value", None)),
        "short_market_value": _as_float(getattr(account, "short_market_value", None)),
        "last_equity": last_equity,
        "day_pl": day_pl,
        "day_pl_pct": day_pl_pct,
    }


def _positions_snapshot(api):
    """Fetch every open Alpaca position with its P/L as plain dicts."""
    raw = _retry(api.get_all_positions, description="alpaca_get_all_positions") or []
    positions = []
    for position in raw:
        positions.append(
            {
                "symbol": getattr(position, "symbol", "?"),
                "qty": _as_float(getattr(position, "qty", None)),
                "avg_entry": _as_float(getattr(position, "avg_entry_price", None)),
                "current_price": _as_float(getattr(position, "current_price", None)),
                "market_value": _as_float(getattr(position, "market_value", None)),
                "unrealized_pl": _as_float(getattr(position, "unrealized_pl", None)),
                "unrealized_plpc": _as_float(getattr(position, "unrealized_plpc", None)),
            }
        )
    return positions


def _filled_orders_snapshot(api, since):
    """Return today's filled orders (i.e. today's closed/open trades)."""
    if GetOrdersRequest is None or QueryOrderStatus is None:
        logger.warning("alpaca-py trading enums unavailable; skipping fills summary")
        return []
    request = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        after=since,
        limit=500,
    )
    orders = (
        _retry(api.get_orders, filter=request, description="alpaca_get_orders") or []
    )
    fills = []
    for order in orders:
        filled_qty = _as_float(getattr(order, "filled_qty", None))
        if filled_qty <= 0:
            continue
        fills.append(
            {
                "id": getattr(order, "id", None),
                "symbol": getattr(order, "symbol", "?"),
                "side": _enum_value(getattr(order, "side", "")).lower(),
                "qty": filled_qty,
                "price": _as_float(getattr(order, "filled_avg_price", None)),
                "status": _enum_value(getattr(order, "status", "")).lower(),
                "filled_at": getattr(order, "filled_at", None),
            }
        )
    return fills


def format_daily_summary(
    date_str,
    account,
    positions,
    fills,
    *,
    trade_stats=None,
    max_positions=20,
    max_fills=20,
    compact=False,
):
    """Render the end-of-day report as text.

    ``compact=True`` produces the Arabic/Darija variant sent to Telegram; the
    default is the verbose English form written to the log. When
    ``trade_stats`` is supplied it adds the day's winning/losing trade counts
    and the realised P/L from :class:`trade_ledger.DailyTradeLedger`.
    """
    lines = []
    if compact:
        lines.append(f"📊 <b>ملخص نهاية اليوم — {_esc(date_str)}</b>")
    else:
        lines.append(f"End-of-day summary for {date_str}")

    # --- Account balance ---------------------------------------------
    if account is None:
        if compact:
            lines.append("⚠️ الحساب: ما قدرناش نجيبو الرصيد من Alpaca")
        else:
            lines.append("Account: balance unavailable (Alpaca account fetch failed)")
    elif compact:
        lines.append(f"💰 <b>رصيد المحفظة:</b> {_money(account['equity'])}")
        lines.append(
            f"• الكاش المتوفر: {_money(account['cash'])} | "
            f"القوة الشرائية: {_money(account['buying_power'])}"
        )
        if account.get("day_pl") is not None:
            pct = account.get("day_pl_pct")
            pct_txt = f" ({pct:+.2f}%)" if pct is not None else ""
            arrow = "📈" if account["day_pl"] >= 0 else "📉"
            lines.append(
                f"{arrow} <b>الربح/الخسارة ديال اليوم:</b> "
                f"{_signed_money(account['day_pl'])}{pct_txt}"
            )
        elif trade_stats and trade_stats.get("trade_count"):
            lines.append(
                f"💵 <b>الربح/الخسارة المحقق ديال اليوم:</b> "
                f"{_signed_money(trade_stats.get('realized_pl'))}"
            )
    else:
        lines.append(
            f"Account: equity {_money(account['equity'])} | "
            f"cash {_money(account['cash'])} | "
            f"buying power {_money(account['buying_power'])} | "
            f"long {_money(account['long_market_value'])} | "
            f"short {_money(account['short_market_value'])}"
        )
        if account.get("day_pl") is not None:
            pct = account.get("day_pl_pct")
            pct_txt = f" ({pct:+.2f}%)" if pct is not None else ""
            lines.append(
                f"  Day P/L: {_signed_money(account['day_pl'])}{pct_txt} "
                f"(prior close {_money(account.get('last_equity'))})"
            )

    # --- Winning vs losing trades ------------------------------------
    if trade_stats is not None:
        if compact:
            lines.append(
                f"✅ <b>صفقات رابحة:</b> {trade_stats.get('wins', 0)} | "
                f"❌ <b>صفقات خاسرة:</b> {trade_stats.get('losses', 0)}"
            )
            extras = []
            if trade_stats.get("breakeven"):
                extras.append(f"➖ متعادلة: {trade_stats['breakeven']}")
            if trade_stats.get("unknown"):
                extras.append(f"⚠️ غير معروفة: {trade_stats['unknown']}")
            if extras:
                lines.append(" | ".join(extras))
            if trade_stats.get("trade_count"):
                lines.append(
                    f"🧾 <b>إجمالي الصفقات المغلقة:</b> "
                    f"{trade_stats['trade_count']} | "
                    f"<b>الربح المحقق:</b> "
                    f"{_signed_money(trade_stats.get('realized_pl'))}"
                )
        else:
            lines.append(
                f"  Trades closed: {trade_stats.get('trade_count', 0)} | "
                f"winners {trade_stats.get('wins', 0)} | "
                f"losers {trade_stats.get('losses', 0)} | "
                f"breakeven {trade_stats.get('breakeven', 0)} | "
                f"unknown {trade_stats.get('unknown', 0)} | "
                f"realised P/L {_signed_money(trade_stats.get('realized_pl'))}"
            )

    # --- Open positions ----------------------------------------------
    if compact:
        lines.append(f"📌 <b>المراكز المفتوحة ({len(positions)}):</b>")
    else:
        lines.append(f"Open positions ({len(positions)}):")
    if not positions:
        lines.append("  • ما كاين حتى مركز مفتوح" if compact else "  none")
    else:
        for position in positions[:max_positions]:
            if compact:
                lines.append(
                    f"  • <b>{_esc(position['symbol'])}</b>: "
                    f"{position['qty']:g} سهم | "
                    f"ثمن الشراء: {_money(position['avg_entry'])} | "
                    f"الثمن الحالي: {_money(position['current_price'])} | "
                    f"الربح/الخسارة: {_signed_money(position['unrealized_pl'])} "
                    f"({position['unrealized_plpc'] * 100.0:+.2f}%)"
                )
            else:
                lines.append(
                    f"  {position['symbol']}: qty {position['qty']:g} | "
                    f"entry {_money(position['avg_entry'])} | "
                    f"last {_money(position['current_price'])} | "
                    f"value {_money(position['market_value'])} | "
                    f"P/L {_signed_money(position['unrealized_pl'])} "
                    f"({position['unrealized_plpc'] * 100.0:+.2f}%)"
                )
        if len(positions) > max_positions:
            lines.append(
                f"  ... و {len(positions) - max_positions} مراكز أخرى"
                if compact
                else f"  ... and {len(positions) - max_positions} more"
            )

    # --- Today's realised trades / filled orders ---------------------
    closed_trades = (trade_stats or {}).get("trades") or []
    if compact:
        if closed_trades:
            lines.append(
                f"🧾 <b>نتائج صفقات اليوم "
                f"({trade_stats.get('trade_count', len(closed_trades))}):</b>"
            )
            for trade in closed_trades[:max_fills]:
                realised = _as_float(trade.get("realized_pl"))
                if realised is None:
                    icon, result = "⚠️", "النتيجة غير معروفة"
                elif realised > 0:
                    icon, result = "✅", _signed_money(realised)
                elif realised < 0:
                    icon, result = "❌", _signed_money(realised)
                else:
                    icon, result = "➖", _signed_money(0.0)
                qty = _as_float(trade.get("qty"), 0.0) or 0.0
                lines.append(
                    f"  {icon} <b>{_esc(trade.get('symbol', '?'))}</b>: "
                    f"{result} ({qty:g} سهم: "
                    f"{_money(trade.get('entry_price'))} ← "
                    f"{_money(trade.get('exit_price'))})"
                )
            if len(closed_trades) > max_fills:
                lines.append(
                    f"  ... و {len(closed_trades) - max_fills} صفقات أخرى"
                )
        else:
            lines.append(f"🧾 <b>الأوامر المنفذة اليوم ({len(fills)}):</b>")
            if not fills:
                lines.append("  • ما كاين حتى أمر منفذ")
            else:
                for fill in fills[:max_fills]:
                    stamp = ""
                    filled_at = fill.get("filled_at")
                    if isinstance(filled_at, datetime):
                        try:
                            stamp = filled_at.astimezone(
                                _market_timezone()
                            ).strftime(" (%H:%M)")
                        except Exception:
                            stamp = ""
                    lines.append(
                        f"  {arabic_action(fill['side'])} "
                        f"{fill['qty']:g} سهم من {_esc(fill['symbol'])} "
                        f"بسعر {_money(fill['price'])}{stamp}"
                    )
                if len(fills) > max_fills:
                    lines.append(
                        f"  ... و {len(fills) - max_fills} أوامر أخرى"
                    )
    else:
        opened = sum(1 for fill in fills if fill["side"] == "buy")
        closed = sum(1 for fill in fills if fill["side"] == "sell")
        lines.append(
            f"Today's filled (closed) orders ({len(fills)} total | "
            f"{opened} buy / {closed} sell):"
        )
        if not fills:
            lines.append("  none")
        else:
            for fill in fills[:max_fills]:
                stamp = ""
                filled_at = fill.get("filled_at")
                if isinstance(filled_at, datetime):
                    try:
                        stamp = filled_at.astimezone(_market_timezone()).strftime(
                            " (%H:%M)"
                        )
                    except Exception:
                        stamp = ""
                lines.append(
                    f"  {fill['side'].upper()} {fill['symbol']} qty {fill['qty']:g} "
                    f"@ {_money(fill['price'])}{stamp} [{fill['status']}]"
                )
            if len(fills) > max_fills:
                lines.append(f"  ... and {len(fills) - max_fills} more")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Order pricing helpers
# --------------------------------------------------------------------------
# Alpaca rejects bracket orders whose stop/take-profit prices are not on a
# valid tick increment (error 42210000, "sub-penny increment does not fulfill
# minimum pricing criteria"). Stocks at/above $1.00 must use whole cents.
_MIN_PRICE_TICK = 0.01


def round_price(value):
    """Round ``value`` to a valid Alpaca price increment (whole cents)."""
    return max(_MIN_PRICE_TICK, round(float(value), 2))


def clamp_stop_loss_pct(value):
    """Clamp a stop distance (``0.0125`` == 1.25%) into the hard band.

    The band is intentionally narrow (1.0%-1.5% by default): every entry must
    be protected by a stop that is neither too tight to survive normal noise
    nor wide enough to produce the oversized losses this guard exists to stop.
    """
    try:
        low, high = sorted((float(STOP_LOSS_MIN_PCT), float(STOP_LOSS_MAX_PCT)))
    except (TypeError, ValueError):
        low, high = 0.010, 0.015
    low = max(0.0, low)
    high = max(low, high)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return high
    if not math.isfinite(numeric):
        return high
    return min(max(numeric, low), high)


def configured_stop_loss_pct():
    """Return ``STOP_LOSS_PCT`` as a distance, clamped to the hard band."""
    try:
        distance = 1.0 - float(STOP_LOSS_PCT)
    except (TypeError, ValueError):
        distance = STOP_LOSS_MAX_PCT
    return clamp_stop_loss_pct(distance)


def stop_loss_price(entry, stop_pct=None):
    """Return a valid broker stop price for an entry at ``entry``.

    The stop distance is clamped to ``[STOP_LOSS_MIN_PCT, STOP_LOSS_MAX_PCT]``
    *after* rounding to a whole cent, so Alpaca can never receive a stop that
    is wider than the configured maximum. Raises :class:`ValueError` when the
    price is too small to express a valid stop (e.g. sub-dollar penny stocks).
    """
    try:
        entry = float(entry)
    except (TypeError, ValueError):
        raise ValueError(f"invalid entry price {entry!r}") from None
    if not math.isfinite(entry) or entry <= _MIN_PRICE_TICK:
        raise ValueError(f"invalid entry price {entry!r}")

    entry = round_price(entry)
    pct = configured_stop_loss_pct() if stop_pct is None else clamp_stop_loss_pct(stop_pct)
    low, high = sorted(
        (max(0.0, float(STOP_LOSS_MIN_PCT)), max(0.0, float(STOP_LOSS_MAX_PCT)))
    )

    stop = round_price(entry * (1.0 - pct))
    if stop >= entry:
        stop = round_price(entry - _MIN_PRICE_TICK)

    # Whole-cent rounding can move the distance slightly outside the band.
    # Nudge it back one tick at a time; a tighter stop is always preferable to
    # an unprotected trade.
    for _ in range(3):
        distance = (entry - stop) / entry
        if distance > high + 1e-9 and stop + _MIN_PRICE_TICK < entry:
            stop = round_price(stop + _MIN_PRICE_TICK)
        elif distance < low - 1e-9 and stop - _MIN_PRICE_TICK > 0:
            stop = round_price(stop - _MIN_PRICE_TICK)
        else:
            break

    distance = (entry - stop) / entry
    if stop <= 0 or stop >= entry or distance < low - 1e-9 or distance > high + 1e-9:
        raise ValueError(
            f"cannot place a {low:.2%}-{high:.2%} stop on entry {entry:.2f} "
            f"(closest valid stop would be {stop:.2f}, "
            f"distance {distance:.2%})"
        )
    return stop


def bracket_prices(price):
    """Return validated ``(stop_loss, take_profit)`` for an entry at ``price``.

    Both legs are rounded to a valid Alpaca increment and must sit strictly on
    the correct side of the entry. The stop loss is always inside the hard
    1.0%-1.5% band. A configuration that would produce an unprotected or
    unbounded trade raises :class:`ValueError` so the caller can skip the
    entry instead of sending it to the broker.
    """
    try:
        entry = float(price)
    except (TypeError, ValueError):
        raise ValueError(f"invalid entry price {price!r}") from None
    if not math.isfinite(entry) or entry <= 0:
        raise ValueError(f"invalid entry price {price!r}")

    entry = round_price(entry)
    stop_loss = stop_loss_price(entry)
    take_profit = round_price(entry * TAKE_PROFIT_PCT)

    if stop_loss >= entry:
        raise ValueError(
            f"stop loss {stop_loss:.2f} is not below entry {entry:.2f} "
            f"(STOP_LOSS_PCT={STOP_LOSS_PCT})"
        )
    if take_profit <= entry:
        raise ValueError(
            f"take profit {take_profit:.2f} is not above entry {entry:.2f} "
            f"(TAKE_PROFIT_PCT={TAKE_PROFIT_PCT})"
        )

    risk = entry - stop_loss
    reward = take_profit - entry
    if risk <= 0 or reward / risk < MIN_REWARD_RISK_RATIO - 1e-9:
        raise ValueError(
            f"reward/risk {reward / risk:.2f} is below the required "
            f"{MIN_REWARD_RISK_RATIO:.2f} (SL={stop_loss:.2f}, "
            f"TP={take_profit:.2f})"
        )
    return stop_loss, take_profit


def position_size(price, cash, equity, stop_loss_price):
    """Return a conservative share count that respects every risk cap.

    The result is the smallest of the risk budget (equity at risk divided by
    the per-share distance to the stop loss), the notional budget (a fraction
    of available cash, with a hard dollar ceiling) and the hard
    ``MAX_SHARES_PER_TRADE`` cap. Returns 0 when any input is unusable, which
    makes the caller skip the trade.
    """
    try:
        entry = float(price)
        available = float(cash)
        portfolio = float(equity)
        stop = float(stop_loss_price)
    except (TypeError, ValueError):
        return 0

    values = (entry, available, portfolio, stop)
    if not all(math.isfinite(value) for value in values):
        return 0
    if entry <= 0 or available <= 0 or portfolio <= 0:
        return 0

    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0

    risk_quantity = portfolio * MAX_PORTFOLIO_RISK_PCT / risk_per_share

    notional_budget = available * POSITION_SIZE_PCT
    if MAX_POSITION_NOTIONAL > 0:
        notional_budget = min(notional_budget, MAX_POSITION_NOTIONAL)
    notional_quantity = notional_budget / entry

    limits = [math.floor(risk_quantity), math.floor(notional_quantity)]
    if MAX_SHARES_PER_TRADE > 0:
        limits.append(MAX_SHARES_PER_TRADE)
    return max(0, min(limits))


def trailing_stop_level(entry, current, high_water=None, existing_stop=None):
    """Return the protective stop that should be active for a position.

    The level only ever moves up. It starts at the entry stop inside the
    1.0%-1.5% band, moves to break-even after the position has *reached* a
    small gain, then trails the highest observed price. Triggering on the
    high-water mark (not the current price) means a pullback from +1.5% to
    +0.2% still keeps the trailing stop armed instead of dropping protection.
    A returned level at or above ``current`` means the position should already
    be closed, which lets the caller act as a software backstop when a broker
    stop cannot be replaced.
    """
    try:
        entry = float(entry)
        current = float(current)
    except (TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(value) for value in (entry, current)):
        return 0.0
    if entry <= 0 or current <= 0:
        return 0.0

    try:
        level = stop_loss_price(entry)
    except ValueError:
        return 0.0

    if existing_stop is not None:
        try:
            existing = float(existing_stop)
        except (TypeError, ValueError):
            existing = 0.0
        if math.isfinite(existing) and existing > 0:
            level = max(level, existing)

    peak = current
    if high_water is not None:
        try:
            candidate = float(high_water)
        except (TypeError, ValueError):
            candidate = current
        if math.isfinite(candidate) and candidate > 0:
            peak = max(peak, candidate)

    gain = (peak - entry) / entry
    if BREAKEVEN_TRIGGER_PCT > 0 and gain >= BREAKEVEN_TRIGGER_PCT:
        level = max(level, entry * (1.0 + max(0.0, BREAKEVEN_BUFFER_PCT)))
    if (
        TRAILING_TRIGGER_PCT > 0
        and gain >= TRAILING_TRIGGER_PCT
        and TRAILING_STOP_PCT > 0
    ):
        level = max(level, peak * (1.0 - max(0.0, TRAILING_STOP_PCT)))

    return round_price(level)


def effective_daily_loss_limit(balance):
    """Return the day's USD loss threshold, or ``None`` when disabled.

    The stricter of the absolute (``-$250``) and percentage (``-0.25%``)
    limits wins, so a smaller account is protected just as tightly as a large
    one.
    """
    if not DAILY_LOSS_LIMIT_ENABLED:
        return None
    limits = []
    if DAILY_LOSS_LIMIT_USD > 0:
        limits.append(float(DAILY_LOSS_LIMIT_USD))
    if DAILY_LOSS_LIMIT_PCT > 0 and balance is not None:
        try:
            numeric = float(balance)
        except (TypeError, ValueError):
            numeric = 0.0
        if math.isfinite(numeric) and numeric > 0:
            limits.append(numeric * float(DAILY_LOSS_LIMIT_PCT))
    if not limits:
        return None
    return min(limits)


# --------------------------------------------------------------------------
# Notification wording helpers (Darija / Arabic)
# --------------------------------------------------------------------------
# Telegram messages are read by humans, not traders, so every template below
# avoids technical jargon and uses clear Darija/Arabic. Messages are sent with
# ``parse_mode="HTML"``; ``<b>`` gives the bold titles from the design while
# all dynamic values are HTML-escaped.
_AR_SENTIMENT = {
    "positive": "الأخبار إيجابية",
    "negative": "الأخبار سلبية",
    "neutral": "الأخبار محايدة",
}


def _esc(value):
    """Escape a dynamic value for Telegram's HTML parse mode."""
    return html.escape(str(value), quote=False)


def _bold(value):
    return f"<b>{_esc(value)}</b>"


def arabic_sentiment(label):
    """Turn a FinBERT label into a clear Arabic/Darija phrase."""
    return _AR_SENTIMENT.get(str(label).strip().lower(), "الأخبار غير واضحة")


def arabic_action(side):
    """Map an order side to the bilingual label used in notifications."""
    return {
        "buy": "شراء (BUY)",
        "sell": "بيع (SELL)",
    }.get(str(side).strip().lower(), "عملية")


def arabic_side(side):
    """Map an order side to the Arabic noun used in problem messages."""
    return {
        "buy": "الشراء",
        "sell": "البيع",
    }.get(str(side).strip().lower(), "التداول")


def plain_quantity(value):
    """Format a share quantity without spurious decimals."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return "?"


def plain_price(value):
    """Format a price as US dollars, tolerating a missing value."""
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$n/a"


def format_buy_message(symbol, price, stop_loss, take_profit, sentiment_label):
    """Build the requested Darija buy alert."""
    return "\n".join(
        [
            f"📈 {_bold(f'شراء (BUY) : {symbol}')}",
            f"• الثمن: {plain_price(price)}",
            f"• {arabic_sentiment(sentiment_label)}",
            f"• حد الخسارة: {plain_price(stop_loss)}",
            f"• حد الربح: {plain_price(take_profit)}",
        ]
    )


def format_sell_message(symbol, price, reason=None):
    """Build the requested Darija sell alert."""
    return "\n".join(
        [
            f"📉 {_bold(f'بيع (SELL) : {symbol}')}",
            f"• الثمن: {plain_price(price)}",
            f"• {_esc(reason or 'السعر وصل لمستويات مرتفعة')}",
        ]
    )


def format_fill_message(symbol, side, quantity, price):
    """Build the Arabic confirmation for a filled order."""
    return "\n".join(
        [
            "✅ <b>تم تنفيذ الأمر</b>",
            f"• العملية: {arabic_action(side)}",
            f"• السهم: {_esc(symbol)}",
            f"• الكمية: {plain_quantity(quantity)} سهم",
            f"• الثمن: {plain_price(price)}",
        ]
    )


def format_session_start_message(when, balance=None):
    """Build the "new trading session" welcome/status message.

    ``when`` is the market-local datetime of the session start and ``balance``
    is the current account equity (or ``None`` when it cannot be fetched).
    """
    if isinstance(when, datetime):
        stamp = when.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    else:
        stamp = str(when)
    balance_text = _money(balance) if balance is not None else "$n/a"
    return "\n".join(
        [
            "🚀 <b>بداية يوم تداول جديد | New Trading Session</b>",
            f"• الوقت: {_esc(stamp)}",
            f"• الرصيد الحالي (Balance): {balance_text}",
            "• البوت شغال وجاهز لمراقبة السوق (Bot is running and active).",
        ]
    )


def format_circuit_breaker_message(limit=None):
    """Build the daily hard-loss (circuit breaker) alert.

    ``limit`` is the effective USD threshold for the day. It defaults to the
    configured absolute limit so the message always matches the requested
    wording (``$250`` by default).
    """
    amount = DAILY_LOSS_LIMIT_USD if limit is None else limit
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        amount = DAILY_LOSS_LIMIT_USD
    return (
        "🚨 <b>توقف حماية الحساب (Circuit Breaker)</b>: "
        f"تجاوزت الخسارة اليومية ${amount:,.0f}. "
        "تم إيقاف التداول اليوم لحماية باقي الرصيد."
    )


def format_cancelled_message(symbol, side):
    """Build the Arabic confirmation for a cancelled order."""
    return "\n".join(
        [
            "❌ <b>تم إلغاء الأمر</b>",
            f"• أمر {arabic_side(side)} ديال {_esc(symbol)} ما تنفذش.",
        ]
    )


def format_rejected_message(symbol, side, error_message=None):
    """Build the Arabic notification for a broker-rejected order."""
    lines = [
        "🚫 <b>تم رفض الأمر</b>",
        f"• أمر {arabic_side(side)} ديال {_esc(symbol)} ترفض من طرف الوسيط.",
    ]
    if error_message:
        lines.append(f"• السبب: {_esc(_truncate(str(error_message), 200))}")
    return "\n".join(lines)


def _order_symbol(order, position=None):
    """Best-effort symbol lookup across lumibot order/position objects."""
    for source in (order, position):
        if source is None:
            continue
        symbol = getattr(source, "symbol", None)
        if symbol:
            return symbol
        asset = getattr(source, "asset", None)
        if isinstance(asset, str) and asset:
            return asset
        symbol = getattr(asset, "symbol", None)
        if symbol:
            return symbol
    return "?"


class MLTraderProMax(Strategy):
    def initialize(self):
        configure_logging()  # idempotent; ensures file handler exists
        self.sleeptime = SLEEPTIME
        self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Guards against producing the end-of-day report twice in one day.
        self._eod_summary_date = None
        # Opening-volatility notice is logged once per market day, not on
        # every iteration of the 30-minute window.
        self._blackout_notice_date = None
        # Ensures the welcome/status message is sent only once per session.
        self._session_message_date = None
        # Persistent FIFO ledger used to report today's winning/losing trades.
        self._trade_ledger = DailyTradeLedger(TRADE_LEDGER_FILE)
        self._ledger_sync_date = None
        # Circuit breaker / trailing-stop state. The risk state file keeps the
        # day's opening equity and the "trading halted" flag across restarts so
        # a reboot cannot silently re-enable trading after the daily limit.
        self._risk_state_path = (
            None if getattr(self, "is_backtesting", False) else RISK_STATE_FILE
        )
        self._risk_state = self._load_risk_state()
        self._high_water = {}
        self._software_stops = {}
        self._closing_symbols = set()
        self._risk_data_stale = False
        self._circuit_breaker_handled = False
        self._last_loss_warning = None
        self._log.info(
            "Strategy initialised (watchlist=%s, workers=%d, feed=%s)",
            WATCHLIST,
            MAX_WORKERS,
            ALPACA_DATA_FEED,
        )
        self._log.info(
            "Risk limits: max %d concurrent positions | max risk/trade=%.2f%% "
            "of equity | max notional=%.2f%% of cash (hard cap %s) | "
            "max shares=%s | SL=%.2f%%-%.2f%% | TP=+%.2f%% | "
            "trailing=%s | daily loss limit=%s | opening filter=%s (%s +%dm)",
            MAX_CONCURRENT_POSITIONS,
            MAX_PORTFOLIO_RISK_PCT * 100.0,
            POSITION_SIZE_PCT * 100.0,
            _money(MAX_POSITION_NOTIONAL) if MAX_POSITION_NOTIONAL > 0 else "off",
            MAX_SHARES_PER_TRADE if MAX_SHARES_PER_TRADE > 0 else "off",
            STOP_LOSS_MIN_PCT * 100.0,
            STOP_LOSS_MAX_PCT * 100.0,
            (TAKE_PROFIT_PCT - 1.0) * 100.0,
            "on" if TRAILING_STOP_ENABLED else "off",
            (
                f"-{_money(DAILY_LOSS_LIMIT_USD)} or "
                f"-{DAILY_LOSS_LIMIT_PCT:.2%} (whichever is stricter)"
                if DAILY_LOSS_LIMIT_ENABLED
                else "off"
            ),
            "on" if TIME_FILTER_ENABLED else "off",
            MARKET_OPEN_TIME,
            OPENING_BLACKOUT_MINUTES,
        )
        # Warm the FinBERT model in the background so the first iteration is fast.
        _alert_pool.submit(finbert_warm_up)
        # Announce the new session as soon as the bot is initialised.
        self._send_session_start_message()

    # ------------------------------------------------------------------
    # Risk state / daily circuit breaker
    # ------------------------------------------------------------------
    def _load_risk_state(self):
        """Load the persisted risk state (empty dict when unavailable).

        A file that exists but cannot be parsed is treated as a tripped
        breaker for the current day: silently forgetting a halt would re-enable
        trading after exactly the kind of crash that most needs the protection.
        """
        path = getattr(self, "_risk_state_path", None)
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
            raise ValueError("risk state is not a JSON object")
        except Exception as exc:
            logger.critical(
                "Risk state %s is unreadable (%s); failing closed and halting "
                "trading for this session",
                path,
                exc,
            )
            return {"_corrupt": True}

    def _save_risk_state(self):
        """Atomically persist the risk state so a restart cannot forget a halt."""
        path = getattr(self, "_risk_state_path", None)
        if not path:
            return
        payload = dict(getattr(self, "_risk_state", {}) or {})
        payload["high_water"] = {
            symbol: round(float(price), 4)
            for symbol, price in (getattr(self, "_high_water", {}) or {}).items()
            if _as_float(price, 0.0) > 0
        }
        payload["software_stops"] = {
            symbol: round(float(price), 4)
            for symbol, price in (getattr(self, "_software_stops", {}) or {}).items()
            if _as_float(price, 0.0) > 0
        }
        directory = os.path.dirname(os.path.abspath(path)) or "."
        tmp_path = None
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".risk_state.", suffix=".tmp", dir=directory
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.warning("Could not save risk state %s: %s", path, exc)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _ensure_risk_day(self, now=None):
        """Return today's risk state, resetting it on the first run of a day."""
        moment = now or self._market_now()
        today = moment.strftime("%Y-%m-%d")
        state = getattr(self, "_risk_state", {})
        if not isinstance(state, dict) or state.get("date") != today:
            # An unreadable state file fails closed: halt this session instead
            # of pretending the day is fresh.
            corrupt = bool(isinstance(state, dict) and state.get("_corrupt"))
            state = {
                "date": today,
                "day_start_equity": None,
                "circuit_breaker_tripped": corrupt,
                "tripped_at": moment.isoformat() if corrupt else None,
                "trip_pnl": None,
                "trip_limit": None,
            }
            if corrupt:
                logger.critical(
                    "Starting %s with a corrupt risk state: trading is halted "
                    "until the next trading day",
                    today,
                )
            self._risk_state = state
            self._high_water = {}
            self._software_stops = {}
            self._closing_symbols = set()
            self._risk_data_stale = False
            self._circuit_breaker_handled = False
            self._last_loss_warning = None
            self._save_risk_state()
        else:
            # Restore trailing state after a restart mid-session.
            stored_high = state.get("high_water")
            if isinstance(stored_high, dict) and not self._high_water:
                self._high_water = {
                    str(symbol).upper(): _as_float(price, 0.0)
                    for symbol, price in stored_high.items()
                    if _as_float(price, 0.0) > 0
                }
            stored_stops = state.get("software_stops")
            if isinstance(stored_stops, dict) and not self._software_stops:
                self._software_stops = {
                    str(symbol).upper(): _as_float(price, 0.0)
                    for symbol, price in stored_stops.items()
                    if _as_float(price, 0.0) > 0
                }
        return state

    def _trading_halted(self, now=None):
        """True when the daily circuit breaker has stopped new entries."""
        if not DAILY_LOSS_LIMIT_ENABLED:
            return False
        state = self._ensure_risk_day(now)
        return bool(state.get("circuit_breaker_tripped"))

    def _position_rows(self, strict=False):
        """Return open positions as plain dicts, with a lumibot fallback.

        ``strict=True`` raises when the broker position list cannot be read.
        Callers that must not fail open (position cap, trailing-stop
        management) use this so a transient API error is never mistaken for
        an empty account.
        """
        api = getattr(getattr(self, "broker", None), "api", None)
        if api is not None:
            try:
                return _positions_snapshot(api)
            except Exception as exc:
                self._log.warning("Could not fetch Alpaca positions: %s", exc)
                if strict:
                    raise

        rows = []
        try:
            positions = self.get_positions() or []
        except Exception as exc:
            self._log.warning("Could not fetch strategy positions: %s", exc)
            if strict:
                raise
            return rows

        for position in positions:
            asset = getattr(position, "asset", None)
            symbol = (
                getattr(asset, "symbol", None)
                or getattr(position, "symbol", None)
                or "?"
            )
            qty = _as_float(
                getattr(position, "quantity", getattr(position, "qty", None)), 0.0
            )
            entry = _as_float(getattr(position, "avg_entry_price", None), 0.0)
            current = _as_float(
                getattr(position, "current_price", None), entry or 0.0
            )
            unrealized = _as_float(getattr(position, "unrealized_pl", None), None)
            if unrealized is None and entry > 0 and current > 0:
                unrealized = (current - entry) * qty
            rows.append(
                {
                    "symbol": str(symbol).upper(),
                    "qty": qty,
                    "avg_entry": entry,
                    "current_price": current,
                    "market_value": _as_float(
                        getattr(position, "market_value", None),
                        current * qty if current and qty else 0.0,
                    ),
                    "unrealized_pl": _as_float(unrealized, 0.0),
                    "unrealized_plpc": _as_float(
                        getattr(position, "unrealized_plpc", None),
                        ((current - entry) / entry) if entry > 0 else 0.0,
                    ),
                }
            )
        return rows

    def _daily_pnl_snapshot(self, now=None):
        """Return realised + unrealised P/L for today and the loss threshold.

        The broker's equity-versus-prior-close figure captures both realised
        and unrealised P/L, including bracket exits that lumibot does not
        report through ``on_filled_order``. The ledger-based figure is used as
        a conservative cross-check, so a missing sell fill cannot hide a loss.
        """
        moment = now or self._market_now()
        today = moment.strftime("%Y-%m-%d")

        account = None
        api = getattr(getattr(self, "broker", None), "api", None)
        if api is not None:
            try:
                account = _account_snapshot(api)
            except Exception as exc:
                self._log.warning("Could not fetch account for daily P/L: %s", exc)

        positions_ok = True
        try:
            positions = self._position_rows(strict=True)
        except Exception as exc:
            self._log.warning("Could not fetch positions for daily P/L: %s", exc)
            positions = []
            positions_ok = False

        equity = _as_float(account.get("equity"), 0.0) if account else 0.0
        if equity <= 0:
            try:
                equity = _as_float(self.get_portfolio_value(), 0.0)
            except Exception as exc:
                self._log.warning("Could not read portfolio value: %s", exc)
                equity = 0.0

        last_equity = _as_float(account.get("last_equity"), 0.0) if account else 0.0
        state = self._ensure_risk_day(moment)
        baseline = _as_float(state.get("day_start_equity"), 0.0)
        baseline_fresh = False
        if baseline <= 0:
            if last_equity > 0:
                baseline = last_equity
            elif equity > 0:
                baseline = equity
            if baseline > 0:
                state["day_start_equity"] = baseline
                self._save_risk_state()
                baseline_fresh = True

        unrealized = sum(
            _as_float(position.get("unrealized_pl"), 0.0) for position in positions
        )
        realized = 0.0
        ledger_ok = True
        ledger = getattr(self, "_trade_ledger", None)
        if ledger is None:
            ledger_ok = False
        else:
            # If the account endpoint is unavailable, import today's fills so
            # bracket exits are not invisible to the daily P/L calculation.
            if account is None and api is not None:
                try:
                    fills = _filled_orders_snapshot(
                        api, _start_of_trading_day(moment)
                    )
                    if fills:
                        ledger.reconcile_fills(fills)
                except Exception as exc:
                    self._log.warning(
                        "Could not reconcile fills for daily P/L: %s", exc
                    )
            try:
                realized = _as_float(ledger.stats(today).get("realized_pl"), 0.0)
            except Exception as exc:
                ledger_ok = False
                self._log.warning("Could not read realised P/L: %s", exc)

        ledger_total = realized + unrealized
        broker_total = None
        if last_equity > 0:
            broker_total = equity - last_equity
        elif baseline > 0 and not baseline_fresh and equity > 0:
            # The account endpoint is unavailable but today's opening equity
            # was recorded earlier, so equity minus that baseline is still a
            # true daily figure.
            broker_total = equity - baseline

        # The broker's equity-versus-prior-close figure is a true *daily*
        # number (realised + unrealised since the open). The ledger figure
        # measures unrealised P/L since entry, so mixing the two would treat
        # yesterday's open loss as today's loss. Use the broker figure
        # whenever it is available and only fall back to the ledger when the
        # account endpoint is unreachable.
        if broker_total is not None:
            total = broker_total
            data_ok = True
        else:
            total = ledger_total
            data_ok = positions_ok and ledger_ok

        return {
            "date": today,
            "realized": realized,
            "unrealized": unrealized,
            "total": total,
            "equity": equity,
            "baseline": baseline,
            "limit": effective_daily_loss_limit(baseline or equity),
            "data_ok": data_ok,
        }

    def _check_circuit_breaker(self, now=None):
        """Trip the daily circuit breaker when the hard loss limit is hit.

        Returns ``True`` when trading is halted for the rest of the session.
        """
        if not DAILY_LOSS_LIMIT_ENABLED:
            return False

        moment = now or self._market_now()
        state = self._ensure_risk_day(moment)

        # A previous process already tripped the breaker today. Make sure any
        # position left behind (crash / manual restart) is liquidated too.
        if state.get("circuit_breaker_tripped"):
            if not self._circuit_breaker_handled:
                self._log.critical(
                    "Circuit breaker already tripped for %s; flattening any "
                    "remaining positions",
                    state.get("date"),
                )
                try:
                    self._liquidate_all_positions("circuit breaker state restored")
                except Exception as exc:
                    logger.error("Could not flatten positions on restore: %s", exc)
                self._circuit_breaker_handled = True
            else:
                # If a liquidation was only partially filled (or a new fill
                # slipped in), keep retrying until the account is flat — but
                # never stack another market sell on top of one still working.
                try:
                    remaining = self._position_rows(strict=True)
                    unknown = False
                except Exception as exc:
                    logger.warning("Could not verify flat account: %s", exc)
                    remaining = None
                    unknown = True
                if unknown or remaining:
                    count = len(remaining or ())
                    if self._has_open_sell_orders():
                        self._log.warning(
                            "Circuit breaker still active with %d open position(s); "
                            "liquidation sells are still working, waiting",
                            count,
                        )
                    else:
                        self._log.critical(
                            "Circuit breaker still active with %d open position(s); "
                            "retrying liquidation",
                            count,
                        )
                        try:
                            self._liquidate_all_positions(
                                "circuit breaker follow-up"
                            )
                        except Exception as exc:
                            logger.error("Follow-up liquidation failed: %s", exc)
            return True

        try:
            snapshot = self._daily_pnl_snapshot(moment)
        except Exception as exc:
            logger.error("Daily P/L check failed: %s", exc, exc_info=True)
            self._risk_data_stale = True
            return False

        if not snapshot.get("data_ok", True):
            if not getattr(self, "_risk_data_stale", False):
                logger.error(
                    "Daily P/L cannot be computed (account and position data "
                    "unavailable); blocking new entries until data is readable"
                )
            self._risk_data_stale = True
            return False

        if getattr(self, "_risk_data_stale", False):
            self._log.warning(
                "Account data is readable again; re-enabling new entries"
            )
            self._risk_data_stale = False

        limit = snapshot.get("limit")
        if limit is None or limit <= 0:
            return False

        total = _as_float(snapshot.get("total"), 0.0)
        if total <= -limit + 1e-9:
            self._trip_circuit_breaker(snapshot, moment)
            return True

        # One warning when the account enters the last quarter of its budget.
        if total <= -0.75 * limit:
            marker = f"{snapshot.get('date')}:{int(limit)}"
            if getattr(self, "_last_loss_warning", None) != marker:
                self._last_loss_warning = marker
                self._log.warning(
                    "Daily P/L %s is close to the %s circuit-breaker limit",
                    _signed_money(total),
                    _money(limit),
                )
        return False

    def _trip_circuit_breaker(self, snapshot, now=None):
        """Persist the halt, liquidate everything and alert Telegram."""
        if getattr(self, "_circuit_breaker_in_progress", False):
            return
        self._circuit_breaker_in_progress = True
        moment = now or self._market_now()
        limit = _as_float(snapshot.get("limit"), DAILY_LOSS_LIMIT_USD)
        total = _as_float(snapshot.get("total"), 0.0)
        try:
            state = self._ensure_risk_day(moment)
            state.update(
                {
                    "circuit_breaker_tripped": True,
                    "tripped_at": moment.isoformat(),
                    "trip_pnl": total,
                    "trip_limit": limit,
                }
            )
            self._save_risk_state()
            self._log.critical(
                "CIRCUIT BREAKER: daily P/L %s breached the %s limit "
                "(realised %s, unrealised %s). Liquidating all positions and "
                "halting new entries for the rest of the day.",
                _signed_money(total),
                _money(limit),
                _signed_money(snapshot.get("realized")),
                _signed_money(snapshot.get("unrealized")),
            )

            try:
                closed = self._liquidate_all_positions("daily loss limit")
                self._log.critical(
                    "Circuit breaker liquidation submitted for %s position(s)",
                    closed,
                )
            except Exception as exc:
                logger.error(
                    "Circuit breaker liquidation failed: %s", exc, exc_info=True
                )

            self._circuit_breaker_handled = True
            self.telegram_send_message(
                format_circuit_breaker_message(limit), wait=True
            )
        finally:
            self._circuit_breaker_in_progress = False

    def _liquidate_all_positions(self, reason):
        """Close every open position and cancel all protective/entry orders."""
        api = getattr(getattr(self, "broker", None), "api", None)
        closed = 0

        if api is not None:
            try:
                _retry(api.cancel_orders, description="cancel_all_orders")
            except Exception as exc:
                self._log.warning("Could not cancel open orders: %s", exc)

            try:
                response = _retry(
                    api.close_all_positions,
                    cancel_orders=True,
                    description="close_all_positions",
                )
                if isinstance(response, list):
                    closed = len(response)
                else:
                    closed = len(self._position_rows())
                self._log.critical(
                    "Liquidated all positions (%s): %d position(s)", reason, closed
                )
                return closed
            except Exception as exc:
                self._log.warning(
                    "close_all_positions failed (%s); closing one by one", exc
                )

            for position in self._position_rows():
                symbol = position.get("symbol")
                try:
                    _retry(
                        api.close_position,
                        symbol,
                        description=f"close_position[{symbol}]",
                    )
                    closed += 1
                except Exception as exc:
                    logger.error("[%s] close_position failed: %s", symbol, exc)
            self._log.critical(
                "Liquidated positions one by one (%s): %d position(s)",
                reason,
                closed,
            )
            return closed

        # Fallback for a broker without the Alpaca trading client (backtests).
        try:
            self.cancel_open_orders()
        except Exception as exc:
            self._log.warning("Could not cancel open orders: %s", exc)

        for position in self._position_rows():
            symbol = position.get("symbol")
            qty = _as_float(position.get("qty"), 0.0)
            if qty == 0:
                continue
            try:
                order = self.create_order(symbol, abs(qty), "sell", type="market")
                self._submit_with_retry(order)
                closed += 1
            except Exception as exc:
                logger.error("[%s] market liquidation failed: %s", symbol, exc)
        self._log.critical(
            "Liquidated positions (%s): %d position(s)", reason, closed
        )
        return closed

    def _has_open_sell_orders(self, symbol=None, market_only=False):
        """True when a sell order is already working (avoid duplicate sells).

        Used before a follow-up liquidation or a market close so an order that
        is still pending is not stacked on top of another one (which could flip
        the account short once both fill). With ``market_only`` the check
        ignores protective stop/take-profit legs, which are expected to exist
        while a position is open.
        """
        target = str(symbol).upper() if symbol else None
        api = getattr(getattr(self, "broker", None), "api", None)

        orders = None
        if (
            api is not None
            and GetOrdersRequest is not None
            and QueryOrderStatus is not None
        ):
            try:
                kwargs = {"status": QueryOrderStatus.OPEN}
                if target:
                    kwargs["symbols"] = [target]
                request = GetOrdersRequest(**kwargs)
                orders = _retry(
                    api.get_orders,
                    filter=request,
                    description="get_open_sell_orders",
                )
            except Exception as exc:
                logger.warning("Could not list open orders: %s", exc)

        if orders is None:
            try:
                orders = self._open_orders()
            except Exception as exc:
                logger.warning("Could not read strategy orders: %s", exc)
                # Unknown state: assume a sell is working so we do not add one.
                return True

        for order in orders or []:
            side = _enum_value(getattr(order, "side", None)).lower()
            status = _enum_value(getattr(order, "status", None)).lower()
            if side != "sell" or status not in self._OPEN_ORDER_STATES:
                continue
            if market_only:
                order_type = _enum_value(
                    getattr(order, "type", None)
                    or getattr(order, "order_type", None)
                ).lower()
                if "market" not in order_type:
                    continue
            order_symbol = str(
                getattr(order, "symbol", None) or _order_symbol(order)
            ).upper()
            if target and order_symbol != target:
                continue
            return True
        return False

    # ------------------------------------------------------------------
    # Data / indicators
    # ------------------------------------------------------------------
    def get_indicators(self, symbol):
        """Return cached RSI/MACD values for ``symbol`` (or ``None``).

        The cache is validated against the latest bar timestamp so values are
        only reused while market data is unchanged (and within the TTL).
        """
        cached = _indicator_cache.get(symbol)
        try:
            # 1-minute data: request enough bars for RSI(14) and MACD(26/9).
            bars = _retry(
                self.get_historical_prices,
                symbol,
                INDICATOR_LOOKBACK,
                "minute",
                description=f"historical_prices[{symbol}]",
            )
            df = getattr(bars, "df", None)

            if df is None or df.empty or len(df) < MIN_BARS_REQUIRED:
                return cached["values"] if cached else None

            last_ts = df.index[-1]
            if cached is not None and cached.get("last_ts") == last_ts:
                return cached["values"]

            values = compute_indicators(df)
            if values is None:
                return None

            _indicator_cache.set(symbol, {"last_ts": last_ts, "values": values})
            return values
        except Exception as exc:
            logger.warning("Indicators failed for %s: %s", symbol, exc)
            return cached["values"] if cached else None

    # ------------------------------------------------------------------
    # Per-symbol analysis (runs inside the thread pool)
    # ------------------------------------------------------------------
    def _analyze_symbol(self, symbol):
        """Gather everything needed to decide on one symbol.

        Each node is isolated so one failure returns ``None`` for this symbol
        only, never propagating to the pool.
        """
        try:
            tech = self.get_indicators(symbol)
        except Exception as exc:
            logger.error("[%s] indicators failed: %s", symbol, exc)
            return None
        if tech is None:
            logger.debug("[%s] not enough data for indicators", symbol)
            return None

        try:
            headlines = fetch_headlines(symbol)
        except Exception as exc:
            logger.error("[%s] news failed: %s", symbol, exc)
            headlines = ""

        try:
            score, label = sentiment_for(symbol, headlines)
        except Exception as exc:
            logger.error("[%s] sentiment failed: %s", symbol, exc)
            score, label = 0.0, "neutral"

        return {
            "symbol": symbol,
            "rsi": tech["rsi"],
            "macd_bull": tech["macd_bull"],
            "price": tech["close"],
            "sentiment_score": score,
            "sentiment_label": label,
        }

    def _analyze_watchlist(self):
        """Analyse every symbol concurrently and return the results."""
        if not WATCHLIST:
            logger.warning("WATCHLIST is empty; nothing to analyse")
            return []

        workers = max(1, min(MAX_WORKERS, len(WATCHLIST)))
        results = []
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="analyse"
        ) as pool:
            futures = {pool.submit(self._analyze_symbol, s): s for s in WATCHLIST}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                except Exception as exc:
                    logger.error("[%s] analysis worker crashed: %s", symbol, exc)
                    continue
                if data is not None:
                    results.append(data)
        return results

    # ------------------------------------------------------------------
    # Order handling (main thread only)
    # ------------------------------------------------------------------
    def _open_orders(self):
        return _retry(
            self.get_orders, description="get_orders"
        ) or []

    def _has_pending_buy(self, symbol) -> bool:
        try:
            open_states = {"pending", "submitted", "new", "accepted", "open"}
            return any(
                order.symbol == symbol
                and order.side == "buy"
                and order.status in open_states
                for order in self._open_orders()
            )
        except Exception as exc:
            logger.warning("[%s] could not read open orders: %s", symbol, exc)
            return False

    def _looks_submitted(self, symbol, side, quantity) -> bool:
        """Best-effort duplicate check used to make order retries idempotent."""
        try:
            for order in self._open_orders():
                if order.symbol != symbol or order.side != side:
                    continue
                try:
                    if abs(float(order.quantity) - float(quantity)) < 1e-9:
                        return True
                except (TypeError, ValueError):
                    continue
        except Exception:
            pass
        return False

    def _submit_with_retry(self, order):
        """Submit an order, retrying transient failures without double-filling.

        Before each retry we check whether a matching order already exists
        (the previous attempt may have succeeded even though the response was
        lost); if so we treat it as done. After a successful submit we wait for
        the broker to acknowledge the order, so a rejected bracket is never
        logged or announced as a filled BUY.
        """

        symbol, side, quantity = order.symbol, order.side, order.quantity

        def _attempt():
            if self._looks_submitted(symbol, side, quantity):
                logger.info(
                    "[%s] matching %s order already present; skipping duplicate submit",
                    symbol,
                    side,
                )
                return None
            return self.submit_order(order)

        result = _retry(
            _attempt,
            description=f"submit_order[{symbol} {side}]",
        )
        if result is not None:
            self._wait_for_order_ack(result)
        failure = self._order_failure(result)
        if failure:
            raise RuntimeError(f"broker did not accept {side} {symbol}: {failure}")
        return result

    def _wait_for_order_ack(self, order):
        """Wait (bounded) for the broker to register ``order``.

        ``lumibot`` exposes an unbounded ``wait_to_be_registered``; using the
        underlying event directly lets the trading loop give up and report a
        rejection instead of hanging forever.
        """
        event = getattr(order, "_new_event", None)
        if event is None:
            return
        timeout = max(0.0, float(ORDER_REGISTRATION_TIMEOUT))
        closed_event = getattr(order, "_closed_event", None)

        def _settled():
            if event.is_set():
                return True
            return bool(closed_event is not None and closed_event.is_set())

        try:
            deadline = time.monotonic() + timeout
            while not _settled():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Poll in short slices so a rejection (which sets only the
                # closed event) is detected without waiting the full timeout.
                event.wait(timeout=min(0.25, remaining))
        except TypeError:  # very old lumibot event without a timeout argument
            return
        except Exception as exc:
            logger.warning(
                "[%s] order acknowledgement wait failed: %s",
                _order_symbol(order),
                exc,
            )
            return
        if not _settled():
            logger.warning(
                "[%s] order %s was not acknowledged within %.1fs",
                _order_symbol(order),
                _enum_value(getattr(order, "status", None)) or "unprocessed",
                timeout,
            )

    @staticmethod
    def _order_failure(order):
        """Return the broker error/status when an order was not accepted."""
        if order is None:
            return None
        error = getattr(order, "_error", None) or getattr(order, "error", None)
        if error:
            return str(error)
        status = _enum_value(getattr(order, "status", None)).lower()
        # ``unprocessed`` after the acknowledgement wait means the broker never
        # registered the order; reporting it as a BUY would be a lie.
        if status in {
            "error",
            "rejected",
            "canceled",
            "cancelled",
            "unprocessed",
        }:
            return status
        return None

    def _create_entry_order(
        self, symbol, quantity, price, stop_loss_price=None, take_profit_price=None
    ):
        """Create a *protected* buy order.

        Every entry is submitted as a bracket carrying both the stop loss and
        the take profit. The stop is never optional: if the broker rejects the
        bracket the exception propagates and the entry is skipped, so the bot
        cannot open an oversized, unprotected position. A market fallback
        exists only behind the explicit ``BRACKET_FALLBACK_TO_MARKET`` opt-in.
        """
        if stop_loss_price is None or take_profit_price is None:
            # Alpaca requires whole-cent increments for stop/limit prices.
            stop_loss_price, take_profit_price = bracket_prices(price)

        # Always bracket an entry that carries protective prices. ``ORDER_TYPE``
        # can no longer silently downgrade a protected trade to a naked market
        # order; that would bypass the strict stop-loss policy.
        order_type = (
            "bracket"
            if (stop_loss_price is not None and take_profit_price is not None)
            else ORDER_TYPE
        )
        try:
            order = self.create_order(
                symbol,
                quantity,
                "buy",
                type=order_type,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )
        except Exception as exc:
            if (
                not REQUIRE_PROTECTIVE_ORDERS
                and ORDER_TYPE == "bracket"
                and BRACKET_FALLBACK_TO_MARKET
            ):
                logger.critical(
                    "[%s] bracket order rejected (%s); submitting an UNPROTECTED "
                    "market order because REQUIRE_PROTECTIVE_ORDERS=false",
                    symbol,
                    exc,
                )
                return self.create_order(symbol, quantity, "buy", type="market")
            if REQUIRE_PROTECTIVE_ORDERS:
                logger.error(
                    "[%s] could not build a protected bracket order (%s); entry "
                    "skipped to avoid an unprotected position",
                    symbol,
                    exc,
                )
            raise

        self._verify_protective_order(
            order, symbol, stop_loss_price, take_profit_price
        )
        return order

    def _verify_protective_order(
        self, order, symbol, stop_loss_price, take_profit_price
    ):
        """Fail closed when the order about to be submitted lacks protection.

        Inspects the actual order object handed to the broker (not just the raw
        request values) so a silently dropped protective leg can never create
        a naked position.
        """
        actual_stop = _as_float(getattr(order, "stop_loss_price", None), 0.0)
        actual_tp = _as_float(getattr(order, "take_profit_price", None), 0.0)

        def _abort(reason):
            try:
                self.cancel_order(order)
                logger.warning("[%s] cancelled unprotected order: %s", symbol, reason)
            except Exception as cancel_exc:
                logger.error(
                    "[%s] could not cancel unprotected order (%s): %s",
                    symbol,
                    reason,
                    cancel_exc,
                )
            raise RuntimeError(f"protective order check failed: {reason}")

        if actual_stop <= 0 or actual_tp <= 0:
            _abort("stop loss and/or take profit missing from the broker order")
        if not actual_stop < actual_tp:
            _abort(
                f"stop loss {actual_stop:.2f} is not below take profit {actual_tp:.2f}"
            )
        if (
            abs(actual_stop - stop_loss_price) > 0.011
            or abs(actual_tp - take_profit_price) > 0.011
        ):
            _abort(
                f"broker levels SL={actual_stop:.2f}/TP={actual_tp:.2f} differ "
                f"from requested SL={stop_loss_price:.2f}/TP={take_profit_price:.2f}"
            )

    # ------------------------------------------------------------------
    # Position protection (break-even / trailing stop)
    # ------------------------------------------------------------------
    _OPEN_ORDER_STATES = {
        "pending",
        "pending_new",
        "pending_replace",
        "submitted",
        "new",
        "accepted",
        "accepted_for_bidding",
        "open",
        "held",
        "partially_filled",
    }

    def _count_exposure_slots(self):
        """Return the number of open positions plus new-entry orders.

        Pending buys for a symbol that is already held do not consume an extra
        slot; every other pending buy does, which prevents the bot from
        submitting a third entry before the first two fill. An unreadable
        position list fails closed (no free slot).
        """
        try:
            rows = self._position_rows(strict=True)
        except Exception as exc:
            logger.warning("Could not count open positions: %s", exc)
            return MAX_CONCURRENT_POSITIONS
        held = {
            str(position.get("symbol") or "").upper()
            for position in rows
            if abs(_as_float(position.get("qty"), 0.0)) > 0
        }
        pending = set()
        try:
            for order in self._open_orders():
                side = _enum_value(getattr(order, "side", None)).lower()
                status = _enum_value(getattr(order, "status", None)).lower()
                if side != "buy" or status not in self._OPEN_ORDER_STATES:
                    continue
                symbol = str(_order_symbol(order)).upper()
                if symbol and symbol != "?" and symbol not in held:
                    pending.add(symbol)
        except Exception as exc:
            # Unknown pending entries: fail closed instead of assuming none.
            logger.warning("Could not count pending entry orders: %s", exc)
            return max(MAX_CONCURRENT_POSITIONS, len(held))
        return len(held | pending)

    def _find_protective_stop(self, symbol, strict=False):
        """Return the best open sell-stop order for ``symbol`` (or ``None``).

        The result is ``{"id", "stop_price", "qty"}``. When several stops
        exist the highest one is used, because that is the level actually
        protecting the position. With ``strict=True`` a failure to read the
        broker's orders raises instead of returning ``None``, so a transient
        API error can never be mistaken for "this position has no stop" and
        trigger a duplicate protective order.
        """
        symbol = str(symbol).upper()
        api = getattr(getattr(self, "broker", None), "api", None)

        orders = None
        if api is not None and GetOrdersRequest is not None and QueryOrderStatus is not None:
            try:
                request = GetOrdersRequest(
                    status=QueryOrderStatus.OPEN, symbols=[symbol]
                )
                orders = _retry(
                    api.get_orders,
                    filter=request,
                    description=f"get_open_orders[{symbol}]",
                )
            except Exception as exc:
                logger.warning("[%s] could not read open orders: %s", symbol, exc)
                if strict:
                    raise

        if orders is None:
            try:
                orders = self._open_orders()
            except Exception as exc:
                logger.warning("[%s] could not read strategy orders: %s", symbol, exc)
                if strict:
                    raise
                return None

        best = self._select_stop_order(orders, symbol)

        # Alpaca returns bracket child legs as separate orders, but lumibot's
        # tracked orders are a useful second source if the API list is
        # incomplete. Only consult them when the broker list had no stop.
        if best is None and api is not None:
            try:
                best = self._select_stop_order(self._open_orders(), symbol)
            except Exception as exc:
                logger.debug(
                    "[%s] strategy-order stop lookup failed: %s", symbol, exc
                )
        return best

    def _select_stop_order(self, orders, symbol):
        """Pick the highest open sell-stop order for ``symbol`` from ``orders``."""
        symbol = str(symbol).upper()
        best = None
        for order in orders or []:
            order_symbol = str(
                getattr(order, "symbol", None) or _order_symbol(order)
            ).upper()
            if order_symbol != symbol:
                continue
            side = _enum_value(getattr(order, "side", None)).lower()
            status = _enum_value(getattr(order, "status", None)).lower()
            order_type = _enum_value(
                getattr(order, "type", None) or getattr(order, "order_type", None)
            ).lower()
            stop_price = _as_float(getattr(order, "stop_price", None), 0.0)
            if side != "sell" or "stop" not in order_type:
                continue
            if status not in self._OPEN_ORDER_STATES or stop_price <= 0:
                continue
            candidate = {
                "id": str(
                    getattr(order, "id", None)
                    or getattr(order, "identifier", None)
                    or ""
                ),
                "stop_price": stop_price,
                "qty": _as_float(
                    getattr(order, "qty", None) or getattr(order, "quantity", None),
                    0.0,
                ),
            }
            if best is None or candidate["stop_price"] > best["stop_price"]:
                best = candidate
        return best

    def _replace_stop_order(self, stop_order, symbol, level):
        """Move an existing broker stop up via Alpaca's replace endpoint."""
        api = getattr(getattr(self, "broker", None), "api", None)
        order_id = (stop_order or {}).get("id")
        if api is None or not order_id or ReplaceOrderRequest is None:
            return False
        new_stop = round_price(level)

        def _attempt():
            return api.replace_order_by_id(
                order_id, ReplaceOrderRequest(stop_price=new_stop)
            )

        try:
            _retry(_attempt, description=f"replace_stop[{symbol}]")
            self._log.info(
                "[%s] trailing stop moved to %s", symbol, plain_price(new_stop)
            )
            return True
        except Exception as exc:
            self._log.warning(
                "[%s] could not move broker stop to %s (%s); using a software "
                "trailing stop instead",
                symbol,
                plain_price(new_stop),
                exc,
            )
            return False

    def _cancel_symbol_orders(self, symbol):
        """Cancel every open order for ``symbol`` (frees shares for a sale).

        Returns the number of cancelled orders, or ``None`` when the order list
        could not be read or an individual cancellation failed. Callers use
        ``None`` to avoid closing a position while a protective leg may still
        be live (a second fill could open a short).
        """
        symbol = str(symbol).upper()
        api = getattr(getattr(self, "broker", None), "api", None)
        cancelled = 0
        failed = 0

        if api is not None and GetOrdersRequest is not None and QueryOrderStatus is not None:
            try:
                request = GetOrdersRequest(
                    status=QueryOrderStatus.OPEN, symbols=[symbol]
                )
                orders = _retry(
                    api.get_orders,
                    filter=request,
                    description=f"get_open_orders[{symbol}]",
                )
            except Exception as exc:
                logger.warning("[%s] could not list orders to cancel: %s", symbol, exc)
                return None
            for order in orders or []:
                order_id = getattr(order, "id", None)
                if not order_id:
                    continue
                try:
                    _retry(
                        api.cancel_order_by_id,
                        order_id,
                        description=f"cancel_order[{symbol}]",
                    )
                    cancelled += 1
                except Exception as exc:
                    failed += 1
                    logger.warning(
                        "[%s] cancel order %s failed: %s", symbol, order_id, exc
                    )
            return None if failed else cancelled

        try:
            for order in self._open_orders():
                if str(_order_symbol(order)).upper() != symbol:
                    continue
                try:
                    self.cancel_order(order)
                    cancelled += 1
                except Exception as exc:
                    failed += 1
                    logger.warning("[%s] cancel order failed: %s", symbol, exc)
        except Exception as exc:
            logger.warning("[%s] could not list orders to cancel: %s", symbol, exc)
            return None
        return None if failed else cancelled

    def _close_position_market(self, symbol, quantity, reason, price=None):
        """Flatten one position at market (used by the software stop fallback)."""
        symbol = str(symbol).upper()
        if symbol in self._closing_symbols:
            return False
        self._closing_symbols.add(symbol)
        try:
            # A market sell that is already working must not be duplicated: two
            # fills would flip the account short.
            if self._has_open_sell_orders(symbol, market_only=True):
                self._log.info(
                    "[%s] a market exit is already working; not submitting another",
                    symbol,
                )
                return False

            # Verify the position before touching its protective legs: if the
            # broker cannot be read we must not cancel the stop and leave the
            # shares naked.
            try:
                rows = self._position_rows(strict=True)
            except Exception as exc:
                logger.warning(
                    "[%s] could not read position (%s); keeping protective "
                    "orders untouched",
                    symbol,
                    exc,
                )
                return False
            live = next(
                (
                    position
                    for position in rows
                    if str(position.get("symbol") or "").upper() == symbol
                ),
                None,
            )
            if live is None:
                self._software_stops.pop(symbol, None)
                return False
            qty = abs(_as_float(live.get("qty"), _as_float(quantity, 0.0)))
            if qty <= 0:
                return False

            api = getattr(getattr(self, "broker", None), "api", None)
            cancelled = self._cancel_symbol_orders(symbol)
            if api is not None and cancelled is None:
                # Without cancelling the bracket legs a market close could leave
                # a live sell-stop behind that later opens a short position.
                logger.warning(
                    "[%s] could not cancel protective orders; refusing to "
                    "close at market until the next cycle",
                    symbol,
                )
                return False

            # Re-read after cancelling: the broker stop may have filled while
            # we were cancelling, and selling a stale quantity would open a
            # short position. Falling back to the pre-cancel quantity is safe
            # because the read succeeded moments ago.
            try:
                refreshed = self._position_rows(strict=True)
            except Exception as exc:
                logger.warning(
                    "[%s] could not re-read position after cancelling orders "
                    "(%s); using the last known quantity",
                    symbol,
                    exc,
                )
                refreshed = None
            if refreshed is not None:
                live = next(
                    (
                        position
                        for position in refreshed
                        if str(position.get("symbol") or "").upper() == symbol
                    ),
                    None,
                )
                if live is None:
                    self._software_stops.pop(symbol, None)
                    return False
                fresh_qty = abs(
                    _as_float(live.get("qty"), _as_float(quantity, 0.0))
                )
                if fresh_qty > 0:
                    qty = fresh_qty

            if api is not None:
                # Server-side close uses the live quantity, so a stop that
                # filled a moment ago cannot turn this into a short sale.
                _retry(
                    api.close_position,
                    symbol,
                    description=f"close_position[{symbol}]",
                )
                quantity_value = qty
            else:
                quantity_value = int(qty) if abs(qty - round(qty)) < 1e-9 else qty
                order = self.create_order(
                    symbol, quantity_value, "sell", type="market"
                )
                self._submit_with_retry(order)

            self._software_stops.pop(symbol, None)
            self._save_risk_state()
            self._log.warning(
                "[%s] closed position (%s): qty=%s",
                symbol,
                reason,
                plain_quantity(quantity_value),
            )
            self.telegram_send_message(
                format_sell_message(
                    symbol,
                    price if price is not None else live.get("current_price"),
                    reason,
                )
            )
            return True
        except Exception as exc:
            logger.error("[%s] market close failed: %s", symbol, exc, exc_info=True)
            return False
        finally:
            self._closing_symbols.discard(symbol)

    def _manage_open_positions(self, now=None):
        """Move stops to break-even/trailing levels and enforce them.

        Runs on every iteration, including the opening blackout: protection
        must never be paused just because new entries are paused.
        """
        try:
            positions = self._position_rows(strict=True)
        except Exception as exc:
            logger.warning(
                "Could not read open positions (%s); skipping stop management "
                "for this cycle and keeping existing protective state",
                exc,
            )
            return
        held = {
            str(position.get("symbol") or "").upper()
            for position in positions
            if abs(_as_float(position.get("qty"), 0.0)) > 0
        }
        # Drop per-symbol trailing state as soon as a position is gone, so a
        # later trade in the same symbol starts from its own entry price.
        for stale in set(self._high_water) - held:
            self._high_water.pop(stale, None)
            self._software_stops.pop(stale, None)
        if not positions:
            return

        for position in positions:
            symbol = str(position.get("symbol") or "").upper()
            qty = _as_float(position.get("qty"), 0.0)
            entry = _as_float(position.get("avg_entry"), 0.0)
            current = _as_float(position.get("current_price"), entry)
            if not symbol or abs(qty) <= 0 or entry <= 0 or current <= 0:
                continue

            high_water = max(
                self._high_water.get(symbol, entry), current
            )
            self._high_water[symbol] = high_water

            stop_order = None
            try:
                stop_order = self._find_protective_stop(symbol, strict=True)
            except Exception as exc:
                logger.warning(
                    "[%s] skipping stop management for this cycle: %s", symbol, exc
                )
                continue
            broker_stop = (
                _as_float(stop_order.get("stop_price"), 0.0) if stop_order else 0.0
            )
            software_stop = _as_float(self._software_stops.get(symbol), 0.0)
            existing = max(broker_stop, software_stop)

            try:
                if TRAILING_STOP_ENABLED:
                    desired = trailing_stop_level(
                        entry, current, high_water, existing
                    )
                else:
                    desired = max(stop_loss_price(entry), existing)
            except Exception as exc:
                logger.warning("[%s] could not compute trailing stop: %s", symbol, exc)
                continue
            if desired <= 0:
                continue

            # Price has reached the trailing level. If the broker already has
            # that stop (or better), let it fill; otherwise close at market.
            if current <= desired:
                if not stop_order or desired > broker_stop + 0.005:
                    self._close_position_market(
                        symbol,
                        qty,
                        f"وصل السعر لحد حماية الأرباح ({plain_price(desired)})",
                        current,
                    )
                continue

            if desired <= existing + 0.005:
                continue

            moved = False
            if stop_order:
                moved = self._replace_stop_order(stop_order, symbol, desired)
            # If no broker stop is visible we deliberately do not submit a
            # second sell order (that could later fill after a take-profit and
            # open a short). The software stop below enforces the same level.

            if moved:
                self._software_stops.pop(symbol, None)
            else:
                # Keep a software level that the loop enforces every minute.
                self._software_stops[symbol] = desired
                self._log.warning(
                    "[%s] broker stop could not be moved; software stop armed "
                    "at %s",
                    symbol,
                    plain_price(desired),
                )
            self._save_risk_state()

    # ------------------------------------------------------------------
    # Signal execution
    # ------------------------------------------------------------------
    def _execute_signal(self, data):
        symbol = data["symbol"]
        price = data["price"]
        rsi = data["rsi"]
        macd_bull = data["macd_bull"]
        label = data["sentiment_label"]

        # No order may leave the bot after the daily circuit breaker fired.
        if self._trading_halted():
            logger.info(
                "[%s] signal ignored: daily circuit breaker has halted trading",
                symbol,
            )
            return

        # Belt-and-braces guard: even if this method is called directly, no
        # order may leave the bot during the opening volatility window.
        if in_opening_blackout(self._market_now()):
            logger.info(
                "[%s] signal ignored: opening volatility filter is active", symbol
            )
            return

        # --- BUY ---------------------------------------------------------
        if rsi < RSI_OVERSOLD and macd_bull and label == "positive":
            # If account/position data could not be read the daily loss guard
            # cannot see the real exposure, so no new risk may be added.
            if getattr(self, "_risk_data_stale", False):
                logger.warning(
                    "[%s] entry skipped: account data is stale, so the daily "
                    "loss limit cannot be verified",
                    symbol,
                )
                return
            if self._count_exposure_slots() >= MAX_CONCURRENT_POSITIONS:
                logger.info(
                    "[%s] entry skipped: maximum %d concurrent positions reached",
                    symbol,
                    MAX_CONCURRENT_POSITIONS,
                )
                return
            if self._has_pending_buy(symbol):
                return
            try:
                pos = _retry(self.get_position, symbol, description="get_position")
            except Exception as exc:
                logger.error("[%s] get_position failed: %s", symbol, exc)
                return
            if pos is not None and getattr(pos, "quantity", 0):
                return  # already holding

            try:
                cash = _retry(self.get_cash, description="get_cash")
            except Exception as exc:
                logger.error("[%s] get_cash failed: %s", symbol, exc)
                return
            try:
                equity = _retry(
                    self.get_portfolio_value, description="get_portfolio_value"
                )
            except Exception as exc:
                logger.warning(
                    "[%s] get_portfolio_value failed (%s); using cash for risk sizing",
                    symbol,
                    exc,
                )
                equity = cash

            # Compute SL/TP before sizing: the distance to the stop determines
            # how many shares the risk budget allows.
            try:
                stop_loss_price, take_profit_price = bracket_prices(price)
            except ValueError as exc:
                logger.error("[%s] refusing entry, invalid SL/TP: %s", symbol, exc)
                self.telegram_send_message(
                    format_rejected_message(symbol, "buy", exc)
                )
                return

            quantity = position_size(
                price, cash, equity, stop_loss_price
            )
            if quantity <= 0:
                self._log.info(
                    "[%s] entry skipped: risk sizing produced 0 shares "
                    "(price=%.2f, cash=%.2f, equity=%.2f, SL=%.2f)",
                    symbol,
                    price,
                    cash,
                    equity,
                    stop_loss_price,
                )
                return

            try:
                order = self._create_entry_order(
                    symbol,
                    quantity,
                    price,
                    stop_loss_price,
                    take_profit_price,
                )
                self._submit_with_retry(order)
                self._log.info(
                    "BUY %s qty=%d @ %.2f (rsi=%.1f, sentiment=%s, SL=%.2f, TP=%.2f)",
                    symbol,
                    quantity,
                    price,
                    rsi,
                    label,
                    stop_loss_price,
                    take_profit_price,
                )
                self.telegram_send_message(
                    format_buy_message(
                        symbol,
                        price,
                        stop_loss_price,
                        take_profit_price,
                        label,
                    )
                )
            except Exception as exc:
                logger.error("[%s] BUY order failed: %s", symbol, exc)
                self.telegram_send_message(
                    format_rejected_message(symbol, "buy", exc)
                )

        # --- SELL --------------------------------------------------------
        elif rsi > RSI_OVERBOUGHT:
            try:
                pos = _retry(self.get_position, symbol, description="get_position")
            except Exception as exc:
                logger.error("[%s] get_position failed: %s", symbol, exc)
                return
            if pos is None or not getattr(pos, "quantity", 0):
                return

            try:
                # Cancel the bracket's stop/take-profit legs first: they hold
                # the shares and would make a separate sell order fail.
                closed = self._close_position_market(
                    symbol,
                    getattr(pos, "quantity", 0),
                    "السعر وصل لمستويات مرتفعة",
                    price,
                )
                if closed:
                    self._log.info(
                        "SELL %s qty=%s @ %.2f (rsi=%.1f)",
                        symbol,
                        pos.quantity,
                        price,
                        rsi,
                    )
            except Exception as exc:
                logger.error("[%s] SELL order failed: %s", symbol, exc)
                self.telegram_send_message(
                    format_rejected_message(symbol, "sell", exc)
                )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _market_now(self):
        """Return the strategy's current market-local, tz-aware time.

        Uses lumibot's ``get_datetime`` so backtests evaluate the simulated
        bar timestamp instead of the wall clock. Falls back to the real clock
        if the data source cannot provide a time.
        """
        try:
            moment = self.get_datetime()
        except Exception as exc:
            logger.warning("get_datetime failed (%s); using system clock", exc)
            moment = None
        if moment is None:
            return datetime.now(_market_timezone())
        if moment.tzinfo is None:
            return moment.replace(tzinfo=_market_timezone())
        return moment.astimezone(_market_timezone())

    def on_trading_iteration(self):
        # Seed any position opened before the ledger existed (once per day).
        self._sync_trade_ledger()

        now = self._market_now()
        self._ensure_risk_day(now)
        # First iteration of a new trading day: announce the session once.
        self._send_session_start_message(now)

        # Risk management runs before anything else and is never skipped by
        # the opening blackout: protecting open positions is always allowed.
        try:
            halted = self._check_circuit_breaker(now)
        except Exception as exc:
            logger.error("Circuit breaker check failed: %s", exc, exc_info=True)
            # Fail closed: if the daily loss cannot be measured, no new risk.
            self._risk_data_stale = True
            halted = False

        # When the breaker is active the liquidation path owns the positions;
        # do not let trailing-stop management race with those market sells.
        if not halted:
            try:
                self._manage_open_positions(now)
            except Exception as exc:
                logger.error(
                    "Position protection (trailing stop) failed: %s",
                    exc,
                    exc_info=True,
                )

        if halted:
            self._log.debug(
                "Trading halted for %s by the daily circuit breaker; no new "
                "entries will be considered",
                now.strftime("%Y-%m-%d"),
            )
            return

        if in_opening_blackout(now):
            today = now.strftime("%Y-%m-%d")
            if self._blackout_notice_date != today:
                start, end = opening_blackout_window(now)
                self._log.info(
                    "Opening volatility filter active (%s-%s): no new trades "
                    "will be submitted during this window",
                    start.strftime("%H:%M"),
                    end.strftime("%H:%M"),
                )
                self._blackout_notice_date = today
            return

        try:
            analyses = self._analyze_watchlist()
        except Exception as exc:
            logger.error("Watchlist analysis failed: %s", exc, exc_info=True)
            return

        for data in analyses:
            try:
                self._execute_signal(data)
            except Exception as exc:
                logger.error(
                    "[%s] execution failed: %s", data.get("symbol"), exc, exc_info=True
                )

        # Catch losses produced during this iteration immediately, rather
        # than waiting for the next one-minute cycle.
        try:
            self._check_circuit_breaker(now)
        except Exception as exc:
            logger.error(
                "Post-execution circuit breaker check failed: %s",
                exc,
                exc_info=True,
            )

    def on_filled_order(self, position, order, price, quantity, multiplier):
        """lumibot lifecycle hook: confirm the fill and feed the trade ledger."""
        try:
            symbol = _order_symbol(order, position)
            side = _enum_value(getattr(order, "side", None))
            filled_qty = quantity if quantity is not None else getattr(
                order, "quantity", None
            )
            fill_id = getattr(order, "identifier", None) or getattr(
                order, "id", None
            )
            self._record_trade_fill(symbol, side, filled_qty, price, fill_id)
            self.telegram_send_message(
                format_fill_message(symbol, side, filled_qty, price)
            )
        except Exception as exc:
            logger.error("on_filled_order failed: %s", exc, exc_info=True)

    def on_canceled_order(self, order):
        """lumibot lifecycle hook: tell the user an order was not completed."""
        try:
            symbol = _order_symbol(order)
            side = _enum_value(getattr(order, "side", None))
            self.telegram_send_message(format_cancelled_message(symbol, side))
        except Exception as exc:
            logger.error("on_canceled_order failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # End-of-day lifecycle (market close)
    # ------------------------------------------------------------------
    def before_market_opens(self):
        """Snapshot opening positions before any order can close them."""
        # A process that stays up across days announces the new session here,
        # before the first trading iteration.
        self._send_session_start_message()
        self._sync_trade_ledger()

    def after_market_closes(self):
        """lumibot lifecycle hook, invoked once after the market closes.

        Produces the daily report, so failures here must never bubble up into
        the executor (which would treat it as a bot crash).
        """
        if not DAILY_SUMMARY_ENABLED:
            self._log.debug(
                "DAILY_SUMMARY_ENABLED is false; skipping end-of-day summary"
            )
            return
        if getattr(self, "is_backtesting", False):
            # Backtests replay many "closes" per run; never hit the live
            # Alpaca account or Telegram from a simulation.
            self._log.debug("Backtesting; skipping end-of-day summary")
            return
        try:
            self.run_after_hours_summary()
        except Exception as exc:
            logger.error("End-of-day summary failed: %s", exc, exc_info=True)

    def run_after_hours_summary(self, reason="the US stock market has closed"):
        """Log today's report and push it to Telegram when configured.

        Safe to call more than once: only the first call per market day does
        any work, so it can also be triggered manually from the trading loop
        if "trading hours have ended" is detected outside the lifecycle hook.
        """
        today = datetime.now(_market_timezone()).strftime("%Y-%m-%d")
        if self._eod_summary_date == today:
            self._log.debug("End-of-day summary already produced for %s", today)
            return
        self._eod_summary_date = today

        # 1. Clear, explicit log message that trading has ended.
        self._log.info(
            "Market closed: %s. Building end-of-day summary for %s.",
            reason,
            today,
        )

        # 2. Fetch and print positions / today's fills / account balance.
        snapshot = self._end_of_day_snapshot()
        ledger = getattr(self, "_trade_ledger", None)
        if ledger is not None:
            try:
                reconciled = ledger.reconcile_fills(snapshot["fills"])
                if reconciled:
                    self._log.info(
                        "Trade ledger reconciled %d fill(s) from Alpaca",
                        reconciled,
                    )
            except Exception as exc:
                self._log.warning("Trade ledger reconciliation failed: %s", exc)
        trade_stats = ledger.stats(today) if ledger is not None else None
        report = format_daily_summary(
            today,
            snapshot["account"],
            snapshot["positions"],
            snapshot["fills"],
            trade_stats=trade_stats,
        )
        # The console handler makes this visible on stdout, and the rotating
        # file handler keeps a copy in logs/.
        self._log.info("Daily performance summary:\n%s", report)

        # 3. Telegram daily summary (only when alerts are configured).
        if not TELEGRAM_ENABLED or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            self._log.info(
                "Telegram is not configured; daily summary kept local only"
            )
            return

        telegram_text = format_daily_summary(
            today,
            snapshot["account"],
            snapshot["positions"],
            snapshot["fills"],
            trade_stats=trade_stats,
            compact=True,
        )
        # Telegram rejects messages longer than 4096 characters. Truncating on
        # line boundaries also keeps the HTML tags balanced.
        self.telegram_send_message(_truncate(telegram_text, 3900), wait=True)
        self._log.info("Daily summary dispatched to Telegram")

    def _sync_trade_ledger(self):
        """Seed the ledger with currently open positions once per day.

        This covers positions opened before the bot (or the ledger file) was
        started, so their eventual sell still counts as a winning/losing trade.
        """
        ledger = getattr(self, "_trade_ledger", None)
        if ledger is None or getattr(self, "is_backtesting", False):
            return

        today = datetime.now(_market_timezone()).strftime("%Y-%m-%d")
        if self._ledger_sync_date == today:
            return

        # Prefer Alpaca positions because they include the average entry price
        # needed for a meaningful P/L when the position is later sold.
        normalized = []
        api = getattr(getattr(self, "broker", None), "api", None)
        if api is not None:
            try:
                normalized = _positions_snapshot(api)
            except Exception as exc:
                self._log.warning(
                    "Could not read Alpaca positions for trade ledger: %s", exc
                )

        if not normalized:
            try:
                positions = self.get_positions() or []
            except Exception as exc:
                self._log.warning(
                    "Could not sync positions to trade ledger: %s", exc
                )
                return
            for position in positions:
                asset = getattr(position, "asset", None)
                symbol = getattr(asset, "symbol", None) or getattr(
                    position, "symbol", None
                )
                normalized.append(
                    {
                        "symbol": symbol,
                        "qty": getattr(position, "quantity", None),
                        "avg_entry": getattr(position, "avg_entry_price", None),
                    }
                )

        seeded = ledger.seed_positions(normalized)
        if seeded:
            self._log.info(
                "Trade ledger seeded existing positions: %s", ", ".join(seeded)
            )
        self._ledger_sync_date = today

    def _record_trade_fill(self, symbol, side, quantity, price, fill_id=None):
        """Record a filled order in the ledger used by the daily report."""
        ledger = getattr(self, "_trade_ledger", None)
        if ledger is None:
            return
        try:
            trade = ledger.record_fill(
                symbol,
                _enum_value(side),
                quantity,
                price,
                fill_id=fill_id,
            )
        except Exception as exc:
            self._log.warning("Could not record fill in trade ledger: %s", exc)
            return

        if trade and trade.get("side") == "sell":
            self._log.info(
                "Trade ledger: %s realised P/L %s",
                trade.get("symbol"),
                _signed_money(trade.get("realized_pl")),
            )

    def _end_of_day_snapshot(self):
        """Collect account, positions and today's fills from Alpaca.

        Each section is fetched independently so a single failing endpoint
        still yields a useful partial report. Never raises.
        """
        snapshot = {"account": None, "positions": [], "fills": []}
        api = getattr(self.broker, "api", None)

        if api is None:  # e.g. a non-Alpaca broker or backtesting
            self._log.warning(
                "Alpaca trading client unavailable; using strategy fallbacks"
            )
            return self._fallback_snapshot()

        try:
            snapshot["account"] = _account_snapshot(api)
        except Exception as exc:
            self._log.warning("Could not fetch Alpaca account balance: %s", exc)

        try:
            snapshot["positions"] = _positions_snapshot(api)
        except Exception as exc:
            self._log.warning("Could not fetch Alpaca positions: %s", exc)

        try:
            snapshot["fills"] = _filled_orders_snapshot(
                api, _start_of_trading_day()
            )
        except Exception as exc:
            self._log.warning("Could not fetch today's Alpaca orders: %s", exc)

        return snapshot

    def _fallback_snapshot(self):
        """Minimal snapshot built from lumibot's tracked state."""
        snapshot = {"account": None, "positions": [], "fills": []}
        try:
            equity = _as_float(self.get_portfolio_value())
            cash = _as_float(self.get_cash())
            snapshot["account"] = {
                "status": "unknown",
                "cash": cash,
                "equity": equity,
                "portfolio_value": equity,
                "buying_power": cash,
                "long_market_value": 0.0,
                "short_market_value": 0.0,
                "last_equity": 0.0,
                "day_pl": None,
                "day_pl_pct": None,
            }
        except Exception as exc:
            self._log.warning("Fallback account lookup failed: %s", exc)

        try:
            for position in self.get_positions() or []:
                asset = getattr(position, "asset", None)
                snapshot["positions"].append(
                    {
                        "symbol": getattr(asset, "symbol", "?"),
                        "qty": _as_float(getattr(position, "quantity", None)),
                        "avg_entry": _as_float(
                            getattr(position, "avg_entry_price", None)
                        ),
                        "current_price": 0.0,
                        "market_value": 0.0,
                        "unrealized_pl": 0.0,
                        "unrealized_plpc": 0.0,
                    }
                )
        except Exception as exc:
            self._log.warning("Fallback position lookup failed: %s", exc)

        return snapshot

    # ------------------------------------------------------------------
    # Alerting (non-blocking, retried on the alert pool)
    # ------------------------------------------------------------------
    def _current_account_balance(self):
        """Return the current account equity, or ``None`` if unavailable.

        Prefers the broker account endpoint (fresh value) and falls back to
        lumibot's cached portfolio value so a status message is still useful
        during a partial API outage.
        """
        api = getattr(getattr(self, "broker", None), "api", None)
        if api is not None:
            try:
                account = _account_snapshot(api)
                balance = account.get("equity")
                if balance is None:
                    balance = account.get("portfolio_value")
                if balance is not None:
                    return balance
            except Exception as exc:
                self._log.warning(
                    "Could not fetch account balance for session message: %s", exc
                )
        try:
            return _as_float(self.get_portfolio_value())
        except Exception as exc:
            self._log.warning(
                "Could not read portfolio value for session message: %s", exc
            )
            return None

    def _send_session_start_message(self, now=None):
        """Send the welcome/status message once per bot start / trading day.

        Called from ``initialize`` for an immediate startup notification and
        from the daily lifecycle/first iteration so a process that stays up
        across midnight announces each new trading day exactly once.
        """
        if not SESSION_START_MESSAGE_ENABLED:
            return
        if getattr(self, "is_backtesting", False):
            # Never hit Alpaca/Telegram from a simulated run.
            return
        if not TELEGRAM_ENABLED or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            self._log.debug(
                "Telegram is not configured; skipping session start message"
            )
            return

        moment = now or self._market_now()
        session_day = moment.strftime("%Y-%m-%d")
        if self._session_message_date == session_day:
            return
        # Claim the day before any I/O so a duplicate call cannot send twice.
        self._session_message_date = session_day

        balance = self._current_account_balance()
        self._log.info(
            "New trading session started (%s); welcome message sent to Telegram "
            "(balance=%s)",
            moment.strftime("%Y-%m-%d %H:%M:%S %Z").strip(),
            _money(balance) if balance is not None else "n/a",
        )
        self.telegram_send_message(format_session_start_message(moment, balance))

    def telegram_send_message(self, message, wait=False):
        """Queue a Telegram alert; ``wait=True`` blocks until it is sent.

        The end-of-day report waits so the message cannot be lost if the
        process exits right after the market-close hook.
        """
        if not TELEGRAM_ENABLED:
            return
        try:
            future = _alert_pool.submit(_send_telegram, message)
            if wait:
                future.result()
        except Exception as exc:
            logger.warning("Telegram dispatch failed: %s", exc)


def _send_telegram(message: str) -> None:
    """Blocking Telegram send (runs on the alert pool). Never raises."""
    if not TELEGRAM_ENABLED or not TELEGRAM_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        call_with_retry(
            requests.post,
            url,
            json=payload,
            timeout=TELEGRAM_TIMEOUT,
            attempts=TELEGRAM_RETRY_ATTEMPTS,
            base_delay=RETRY_BASE_DELAY,
            max_delay=RETRY_MAX_DELAY,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            jitter=RETRY_JITTER,
            description="telegram_send_message",
        )
    except Exception as exc:
        logger.warning("Telegram message dropped after retries: %s", exc)


def main():
    """Entry point: build the broker/trader and run the strategy."""
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        logger.error("ALPACA_API_KEY / ALPACA_API_SECRET are not set; aborting.")
        raise SystemExit(2)

    if IS_BACKTESTING:
        from lumibot.backtesting import YahooDataBacktesting # pyright: ignore[reportMissingImports]

        start = datetime.fromisoformat(BACKTEST_START)
        end = datetime.fromisoformat(BACKTEST_END)
        logger.info("Starting BACKTEST from %s to %s", start, end)
        MLTraderProMax.backtest(YahooDataBacktesting, start, end)
        return

    logger.info("Starting LIVE/PAPER trading (paper=%s)", alpaca_config()["is_paper"])
    config = alpaca_config()
    # Pin every historical/latest stock request to the configured feed (IEX by
    # default); lumibot's built-in AlpacaData would otherwise default to SIP.
    data_source = AlpacaIEXData(config)
    broker = Alpaca(config, data_source=data_source)
    trader = Trader()
    strategy = MLTraderProMax(broker=broker)
    trader.add_strategy(strategy)

    try:
        trader.run_all()
    except KeyboardInterrupt:
        logger.info("Interrupted by user; shutting down.")
    except Exception as exc:
        logger.critical("Trader crashed: %s", exc, exc_info=True)
        raise
    finally:
        _alert_pool.shutdown(wait=False)


if __name__ == "__main__":
    main()
