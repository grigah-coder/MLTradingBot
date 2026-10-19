"""Centralised configuration.

Every tunable lives here and can be overridden through environment variables
(loaded from ``.env`` via python-dotenv), so nothing operational is hardcoded
in the trading logic.
"""

import logging
import os

from dotenv import load_dotenv # pyright: ignore[reportMissingImports]

load_dotenv()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def _env_str(name, default=""):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    try:
        return int(_env_str(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(_env_str(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_list(name, default):
    value = os.getenv(name)
    if not value:
        return list(default)
    items = [item.strip().upper() for item in value.replace(";", ",").split(",")]
    return [item for item in items if item] or list(default)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN = _env_str("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = _env_str("TELEGRAM_CHAT_ID")
ALPACA_API_KEY = _env_str("ALPACA_API_KEY")
ALPACA_API_SECRET = _env_str("ALPACA_API_SECRET")
ALPACA_PAPER = _env_bool("ALPACA_PAPER", True)
ALPACA_DATA_FEED = _env_str("ALPACA_DATA_FEED", "iex")  # iex | sip

# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------
_DEFAULT_WATCHLIST = [
    "AAPL",
    "NVDA",
    "TSLA",
    "MSFT",
    "AMZN",
    "META",
    "GOOGL",
    "NFLX",
    "AMD",
    "QQQ",
]
WATCHLIST = _env_list("WATCHLIST", _DEFAULT_WATCHLIST)

# ---------------------------------------------------------------------------
# Strategy / risk
# ---------------------------------------------------------------------------
SLEEPTIME = _env_str("SLEEPTIME", "1M")
CASH_AT_RISK = _env_float("CASH_AT_RISK", 0.5)
LIMIT_ORDER_OFFSET = _env_float("LIMIT_ORDER_OFFSET", 0.001)

RSI_PERIOD = _env_int("RSI_PERIOD", 14)
RSI_OVERSOLD = _env_float("RSI_OVERSOLD", 40)
RSI_OVERBOUGHT = _env_float("RSI_OVERBOUGHT", 70)
MACD_FAST = _env_int("MACD_FAST", 12)
MACD_SLOW = _env_int("MACD_SLOW", 26)
MACD_SIGNAL = _env_int("MACD_SIGNAL", 9)

POSITION_SIZE_PCT = _env_float("POSITION_SIZE_PCT", 0.05)  # fraction of cash
STOP_LOSS_PCT = _env_float("STOP_LOSS_PCT", 0.98)          # -2%
TAKE_PROFIT_PCT = _env_float("TAKE_PROFIT_PCT", 1.04)      # +4%

# Order handling. Modern lumibot uses ``type=`` (e.g. "bracket"); when a
# bracket order cannot be placed we optionally fall back to a plain market
# order so the trade is not silently dropped.
ORDER_TYPE = _env_str("ORDER_TYPE", "bracket")
BRACKET_FALLBACK_TO_MARKET = _env_bool("BRACKET_FALLBACK_TO_MARKET", True)

# ---------------------------------------------------------------------------
# Data / indicators
# ---------------------------------------------------------------------------
INDICATOR_LOOKBACK = _env_int("INDICATOR_LOOKBACK", 200)  # 1-minute bars
MIN_BARS_REQUIRED = _env_int("MIN_BARS_REQUIRED", 30)
NEWS_LIMIT = _env_int("NEWS_LIMIT", 5)

# ---------------------------------------------------------------------------
# Caching (seconds / entries)
# ---------------------------------------------------------------------------
NEWS_CACHE_TTL = _env_float("NEWS_CACHE_TTL", 300)
SENTIMENT_CACHE_TTL = _env_float("SENTIMENT_CACHE_TTL", 300)
INDICATOR_CACHE_TTL = _env_float("INDICATOR_CACHE_TTL", 30)
CACHE_MAXSIZE = _env_int("CACHE_MAXSIZE", 512)

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
MAX_WORKERS = _env_int("MAX_WORKERS", min(8, max(1, len(WATCHLIST))))
ALERT_WORKERS = _env_int("ALERT_WORKERS", 2)

# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------
RETRY_ATTEMPTS = _env_int("RETRY_ATTEMPTS", 4)
RETRY_BASE_DELAY = _env_float("RETRY_BASE_DELAY", 1.0)
RETRY_MAX_DELAY = _env_float("RETRY_MAX_DELAY", 30.0)
RETRY_BACKOFF_FACTOR = _env_float("RETRY_BACKOFF_FACTOR", 2.0)
RETRY_JITTER = _env_bool("RETRY_JITTER", True)

# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------
TELEGRAM_ENABLED = _env_bool("TELEGRAM_ENABLED", True)
TELEGRAM_TIMEOUT = _env_float("TELEGRAM_TIMEOUT", 5)
TELEGRAM_RETRY_ATTEMPTS = _env_int("TELEGRAM_RETRY_ATTEMPTS", 3)

# ---------------------------------------------------------------------------
# End-of-day summary
# ---------------------------------------------------------------------------
# When enabled, an end-of-day report (open positions, today's fills and the
# account balance) is logged and pushed to Telegram right after the market
# closes. MARKET_TIMEZONE defines what "today" means for the report.
DAILY_SUMMARY_ENABLED = _env_bool("DAILY_SUMMARY_ENABLED", True)
MARKET_TIMEZONE = _env_str("MARKET_TIMEZONE", "America/New_York")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = _env_str("LOG_LEVEL", "INFO").upper()
LOG_DIR = _env_str("LOG_DIR", "logs")
LOG_FILE = _env_str("LOG_FILE", "tradingbot.log")
LOG_MAX_BYTES = _env_int("LOG_MAX_BYTES", 5_000_000)
LOG_BACKUP_COUNT = _env_int("LOG_BACKUP_COUNT", 5)

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
IS_BACKTESTING = _env_bool("IS_BACKTESTING", False)
BACKTEST_START = _env_str("BACKTEST_START", "2024-01-01")
BACKTEST_END = _env_str("BACKTEST_END", "2024-01-10")


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
def alpaca_config():
    """Build the lumibot Alpaca config dict from environment settings."""
    return {
        "API_KEY": ALPACA_API_KEY,
        "API_SECRET": ALPACA_API_SECRET,
        "is_paper": ALPACA_PAPER,
        "FEED": ALPACA_DATA_FEED,
    }


def resolved_log_level():
    """Return ``LOG_LEVEL`` as a :mod:`logging` level constant."""
    return getattr(logging, str(LOG_LEVEL).upper(), logging.INFO)
