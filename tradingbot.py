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
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader

from cache_utils import TTLCache
from finbert_utils import estimate_sentiment, warm_up as finbert_warm_up
from logging_setup import configure_logging
from retry_utils import call_with_retry
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


class MLTraderProMax(Strategy):
    def initialize(self):
        configure_logging()  # idempotent; ensures file handler exists
        self.sleeptime = SLEEPTIME
        self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
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
        stop_loss_price = price * STOP_LOSS_PCT
        take_profit_price = price * TAKE_PROFIT_PCT
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
                order = self._create_entry_order(symbol, quantity, price)
                self._submit_with_retry(order)
                self._log.info(
                    "BUY %s qty=%d @ %.2f (rsi=%.1f, sentiment=%s, SL=%.2f, TP=%.2f)",
                    symbol,
                    quantity,
                    price,
                    rsi,
                    label,
                    price * STOP_LOSS_PCT,
                    price * TAKE_PROFIT_PCT,
                )
                self.telegram_send_message(
                    f"✅ BUY {symbol} @ {price:.2f} (Sentiment: {label}, "
                    f"SL: {price * STOP_LOSS_PCT:.2f}, "
                    f"TP: {price * TAKE_PROFIT_PCT:.2f})"
                )
            except Exception as exc:
                logger.error("[%s] BUY order failed: %s", symbol, exc)

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
                self.telegram_send_message(
                    f"💰 SELL {symbol} @ {price:.2f} (RSI > {RSI_OVERBOUGHT:.0f})"
                )
            except Exception as exc:
                logger.error("[%s] SELL order failed: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def on_trading_iteration(self):
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

    def on_order_event(self, order_event):
        try:
            order = order_event.order
            logger.info(
                "Order event: %s %s -> %s",
                order.symbol,
                order.side.upper(),
                order_event.status,
            )
            if order_event.status == "filled":
                self.telegram_send_message(
                    f"📈 FILLED: {order.symbol} {order.side.upper()} "
                    f"{order.quantity} @ {order.average_fill_price}"
                )
            elif order_event.status == "cancelled":
                self.telegram_send_message(
                    f"❌ CANCELLED: {order.symbol} {order.side.upper()}"
                )
            elif order_event.status == "rejected":
                self.telegram_send_message(
                    f"🚫 REJECTED: {order.symbol} {order.side.upper()} - "
                    f"{order.error_message}"
                )
        except Exception as exc:
            logger.error("on_order_event failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Alerting (non-blocking, retried on the alert pool)
    # ------------------------------------------------------------------
    def telegram_send_message(self, message):
        if not TELEGRAM_ENABLED:
            return
        try:
            _alert_pool.submit(_send_telegram, message)
        except Exception as exc:
            logger.warning("Telegram dispatch failed: %s", exc)


def _send_telegram(message: str) -> None:
    """Blocking Telegram send (runs on the alert pool). Never raises."""
    if not TELEGRAM_ENABLED or not TELEGRAM_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
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
        from lumibot.backtesting import YahooDataBacktesting

        start = datetime.fromisoformat(BACKTEST_START)
        end = datetime.fromisoformat(BACKTEST_END)
        logger.info("Starting BACKTEST from %s to %s", start, end)
        MLTraderProMax.backtest(YahooDataBacktesting, start, end)
        return

    logger.info("Starting LIVE/PAPER trading (paper=%s)", alpaca_config()["is_paper"])
    broker = Alpaca(alpaca_config())
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
