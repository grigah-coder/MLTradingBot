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
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
    from alpaca.trading.requests import GetOrdersRequest # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    QueryOrderStatus = None  # type: ignore
    GetOrdersRequest = None  # type: ignore
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
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    ORDER_TYPE,
    BRACKET_FALLBACK_TO_MARKET,
    TELEGRAM_ENABLED,
    TELEGRAM_TIMEOUT,
    TELEGRAM_RETRY_ATTEMPTS,
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


def bracket_prices(price):
    """Return ``(stop_loss, take_profit)`` on valid cent increments."""
    return (
        round_price(price * STOP_LOSS_PCT),
        round_price(price * TAKE_PROFIT_PCT),
    )


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
        # Persistent FIFO ledger used to report today's winning/losing trades.
        self._trade_ledger = DailyTradeLedger(TRADE_LEDGER_FILE)
        self._ledger_sync_date = None
        self._log.info(
            "Strategy initialised (watchlist=%s, workers=%d, feed=%s)",
            WATCHLIST,
            MAX_WORKERS,
            ALPACA_DATA_FEED,
        )
        # Warm the FinBERT model in the background so the first iteration is fast.
        _alert_pool.submit(finbert_warm_up)

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
        lost); if so we treat it as done.
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

        return _retry(
            _attempt,
            description=f"submit_order[{symbol} {side}]",
        )

    def _create_entry_order(self, symbol, quantity, price):
        """Create the buy order, falling back from bracket to market if needed."""
        # Alpaca requires whole-cent increments for stop/limit prices.
        stop_loss_price, take_profit_price = bracket_prices(price)
        try:
            return self.create_order(
                symbol,
                quantity,
                "buy",
                type=ORDER_TYPE,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )
        except Exception as exc:
            if ORDER_TYPE == "bracket" and BRACKET_FALLBACK_TO_MARKET:
                logger.warning(
                    "[%s] bracket order rejected (%s); falling back to market order",
                    symbol,
                    exc,
                )
                return self.create_order(symbol, quantity, "buy", type="market")
            raise

    def _execute_signal(self, data):
        symbol = data["symbol"]
        price = data["price"]
        rsi = data["rsi"]
        macd_bull = data["macd_bull"]
        label = data["sentiment_label"]

        # --- BUY ---------------------------------------------------------
        if rsi < RSI_OVERSOLD and macd_bull and label == "positive":
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

            quantity = int((cash * POSITION_SIZE_PCT) / price) if price else 0
            if quantity <= 0:
                return

            try:
                stop_loss_price, take_profit_price = bracket_prices(price)
                order = self._create_entry_order(symbol, quantity, price)
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
                order = self.create_order(
                    symbol, pos.quantity, "sell", type="market"
                )
                self._submit_with_retry(order)
                self._log.info(
                    "SELL %s qty=%s @ %.2f (rsi=%.1f)", symbol, pos.quantity, price, rsi
                )
                self.telegram_send_message(format_sell_message(symbol, price))
            except Exception as exc:
                logger.error("[%s] SELL order failed: %s", symbol, exc)
                self.telegram_send_message(
                    format_rejected_message(symbol, "sell", exc)
                )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def on_trading_iteration(self):
        # Seed any position opened before the ledger existed (once per day).
        self._sync_trade_ledger()
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
