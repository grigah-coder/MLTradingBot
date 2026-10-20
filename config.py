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


_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _project_path(value):
    """Resolve ``value`` against the project directory when it is relative.

    Persistent risk state must not depend on the process working directory:
    starting the bot from another folder would otherwise lose track of a
    tripped circuit breaker.
    """
    return value if os.path.isabs(value) else os.path.join(_PROJECT_DIR, value)


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

# Opening volatility filter. The first minutes after the cash open are the
# most erratic of the day, so no strategy order is submitted between the open
# and ``OPENING_BLACKOUT_MINUTES`` later. Times are market-local, meaning
# 09:30-10:00 New York time (EST/EDT handled automatically by the timezone).
TIME_FILTER_ENABLED = _env_bool("TIME_FILTER_ENABLED", True)
MARKET_OPEN_TIME = _env_str("MARKET_OPEN_TIME", "09:30")
OPENING_BLACKOUT_MINUTES = _env_int("OPENING_BLACKOUT_MINUTES", 30)

# Position sizing. Each entry is capped by *all* of these limits:
#   * MAX_PORTFOLIO_RISK_PCT  - equity at risk if the stop loss is hit,
#   * POSITION_SIZE_PCT       - maximum notional as a fraction of cash,
#   * MAX_POSITION_NOTIONAL   - hard dollar cap per position,
#   * MAX_SHARES_PER_TRADE    - hard share cap (0 disables it).
# Risk-based sizing uses the distance to the stop, so a wider stop
# automatically buys fewer shares.
POSITION_SIZE_PCT = _env_float("POSITION_SIZE_PCT", 0.03)  # max 3% of cash
MAX_POSITION_NOTIONAL = _env_float("MAX_POSITION_NOTIONAL", 2500.0)  # $ cap
MAX_PORTFOLIO_RISK_PCT = _env_float("MAX_PORTFOLIO_RISK_PCT", 0.01)  # 1% of equity
MAX_SHARES_PER_TRADE = _env_int("MAX_SHARES_PER_TRADE", 100)  # 0 = no cap

# Hard cap on the number of positions that may be open at the same time.
MAX_CONCURRENT_POSITIONS = _env_int("MAX_CONCURRENT_POSITIONS", 2)

# Stop loss is always attached to the entry (bracket order) and its distance
# is clamped into this hard band, whatever STOP_LOSS_PCT says. STOP_LOSS_PCT
# keeps the historic multiplier form (0.985 == -1.5%).
STOP_LOSS_MIN_PCT = _env_float("STOP_LOSS_MIN_PCT", 0.010)  # 1.0% minimum
STOP_LOSS_MAX_PCT = _env_float("STOP_LOSS_MAX_PCT", 0.015)  # 1.5% maximum
STOP_LOSS_PCT = _env_float("STOP_LOSS_PCT", 0.985)          # -1.5%
TAKE_PROFIT_PCT = _env_float("TAKE_PROFIT_PCT", 1.04)       # +4%
# Entries whose take-profit distance is smaller than this multiple of the
# stop-loss distance are rejected instead of being sent to the broker.
MIN_REWARD_RISK_RATIO = _env_float("MIN_REWARD_RISK_RATIO", 1.0)

# Trailing / break-even stop. Once a position has reached a
# BREAKEVEN_TRIGGER_PCT gain the stop moves to entry + BREAKEVEN_BUFFER_PCT;
# once it has reached TRAILING_TRIGGER_PCT the stop trails TRAILING_STOP_PCT
# behind the highest price seen. The stop only ever moves up.
TRAILING_STOP_ENABLED = _env_bool("TRAILING_STOP_ENABLED", True)
BREAKEVEN_TRIGGER_PCT = _env_float("BREAKEVEN_TRIGGER_PCT", 0.005)  # +0.5%
BREAKEVEN_BUFFER_PCT = _env_float("BREAKEVEN_BUFFER_PCT", 0.001)    # +0.1%
TRAILING_TRIGGER_PCT = _env_float("TRAILING_TRIGGER_PCT", 0.010)    # +1.0%
TRAILING_STOP_PCT = _env_float("TRAILING_STOP_PCT", 0.008)          # 0.8%

# Daily hard loss limit (circuit breaker). The effective threshold is the
# stricter of the two limits: -$250 or -0.25% of the day's opening equity.
# When hit, every position is liquidated and no new trade is opened until the
# next trading day.
DAILY_LOSS_LIMIT_ENABLED = _env_bool("DAILY_LOSS_LIMIT_ENABLED", True)
DAILY_LOSS_LIMIT_USD = _env_float("DAILY_LOSS_LIMIT_USD", 250.0)
DAILY_LOSS_LIMIT_PCT = _env_float("DAILY_LOSS_LIMIT_PCT", 0.0025)  # 0.25%
RISK_STATE_FILE = _project_path(
    _env_str(
        "RISK_STATE_FILE",
        os.path.join(_env_str("LOG_DIR", "logs"), "risk_state.json"),
    )
)

# Order handling. Modern lumibot uses ``type=`` (e.g. "bracket").
# Entries are always submitted as brackets with a stop loss and take profit.
# ``REQUIRE_PROTECTIVE_ORDERS`` is kept for backwards compatibility; the
# verification step is also applied when it is false, so an entry can never
# silently lose its protective legs. The only way to allow a naked market
# order is the explicit ``BRACKET_FALLBACK_TO_MARKET`` opt-in below.
ORDER_TYPE = _env_str("ORDER_TYPE", "bracket")
REQUIRE_PROTECTIVE_ORDERS = _env_bool("REQUIRE_PROTECTIVE_ORDERS", True)
BRACKET_FALLBACK_TO_MARKET = _env_bool("BRACKET_FALLBACK_TO_MARKET", False)
# Maximum time to wait for the broker to acknowledge a submitted order. When
# the deadline passes the order is treated as rejected, so a refused bracket
# can never be reported as a successful BUY.
ORDER_REGISTRATION_TIMEOUT = _env_float("ORDER_REGISTRATION_TIMEOUT", 15.0)

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
# Send one "new trading session" welcome message per bot start / trading day.
SESSION_START_MESSAGE_ENABLED = _env_bool("SESSION_START_MESSAGE_ENABLED", True)

# ---------------------------------------------------------------------------
# End-of-day summary
# ---------------------------------------------------------------------------
# When enabled, an end-of-day report (daily P/L, winning/losing trades, open
# positions and the account balance) is logged and pushed to Telegram right
# after the market closes. MARKET_TIMEZONE defines what "today" means for the
# report. TRADE_LEDGER_FILE stores open lots and realised trade outcomes so
# the win/loss counts survive a restart.
DAILY_SUMMARY_ENABLED = _env_bool("DAILY_SUMMARY_ENABLED", True)
MARKET_TIMEZONE = _env_str("MARKET_TIMEZONE", "America/New_York")
TRADE_LEDGER_FILE = _env_str(
    "TRADE_LEDGER_FILE",
    os.path.join(_env_str("LOG_DIR", "logs"), "trade_ledger.json"),
)

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
