# 🚀 MLTraderProMax

An asynchronous, production-ready automated algorithmic trading bot built in Python. It integrates **Lumibot** for broker execution (Alpaca), **FinBERT** for real-time financial news sentiment analysis, and multi-threaded processing for lightning-fast performance.

---

## 🛠️ Key Features
- **Concurrent Execution:** Processes the entire asset watchlist in parallel using thread pools.
- **Sentiment Analysis:** Evaluates recent financial headlines via Hugging Face's **FinBERT** transformer model and requires a minimum confidence before a news label can trigger a trade.
- **5-Minute Indicators:** RSI, MACD and the SMA/EMA trend stack are computed from 5-minute bars, which filters out most 1-minute noise.
- **Strict Entry Confirmation:** A buy is sent only when the SMA and EMA stacks are bullish, MACD is bullish, RSI is at or below the oversold threshold and FinBERT sentiment is positive with enough confidence. One disagreeing input keeps the bot flat.
- **TTL Caching:** Built-in in-memory caching for news, sentiment scores, and technical indicators to minimize redundant API calls and processing overhead.
- **Resilient Error Handling:** Robust exponential backoff with jitter for network failures and Alpaca rate limits.
- **Automated Alerts:** Dispatches instant status updates (fills, buy/sell actions and genuine user/strategy cancellations) directly to **Telegram** in clear **Darija/Arabic**, including a once-per-session startup message with the current balance. Automated bracket/OCO and protective-leg cancellations are logged silently instead of paging the user.
- **Order-State Tracking:** A thread-safe tracker keeps the live stop-loss/take-profit ids in sync with Alpaca. Filled, cancelled and expired ids are purged immediately, and replaced stop orders are re-registered under their new Alpaca id, so the bot never retries a closed order.
- **End-of-Day Report:** When the market closes, logs and sends an Arabic daily summary with daily profit/loss ($ and %), winning vs losing trades, the portfolio balance and open positions.
- **Risk Management:** A hard 1.0%-1.5% stop loss on every entry, break-even/trailing stops that protect profits, a maximum of 2 concurrent positions, a $2,500 (or 3% of cash) cap per trade, a 15-minute minimum hold before a signal exit, an opening-volatility time filter (no entries 09:30–10:00 ET), and a daily -$250 / -0.25% circuit breaker that liquidates the account and halts trading for the day. Broker-side stop-loss and take-profit exits always apply, even during the minimum hold.

---

## 📂 Project Structure
```text
MLTradingBot/
│
├── tradingbot.py         # Main entry point and strategy loop
├── config.py             # Centralised environment and strategy configuration
├── order_state.py        # Thread-safe stop-loss/take-profit order-id tracker
├── trade_ledger.py       # Persistent FIFO ledger for daily win/loss reporting
├── finbert_utils.py      # FinBERT NLP sentiment pipeline and batch inference
├── cache_utils.py        # Thread-safe in-memory TTL cache implementation
├── logging_setup.py      # Size-bounded rotating file and console logger
├── retry_utils.py        # Exponential backoff with jitter helper
├── requirements.txt      # Project dependencies
├── tests/                # Risk-control regression tests
└── .gitignore            # Ignored system and cache files
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/grigah-coder/MLTradingBot.git
   cd MLTradingBot
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure your environment variables:**  
   Create a `.env` file in the root directory and populate your keys:
   ```env
   ALPACA_API_KEY=your_alpaca_api_key
   ALPACA_API_SECRET=your_alpaca_api_secret
   ALPACA_PAPER=True
   TELEGRAM_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   # Optional: end-of-day report controls
   DAILY_SUMMARY_ENABLED=True
   MARKET_TIMEZONE=America/New_York
   TRADE_LEDGER_FILE=logs/trade_ledger.json
   # Optional: indicator / signal controls (defaults shown)
   INDICATOR_TIMEFRAME=5 minutes
   INDICATOR_LOOKBACK=200
   MIN_BARS_REQUIRED=60
   RSI_PERIOD=14
   RSI_OVERSOLD=40
   TREND_SMA_FAST=20
   TREND_SMA_SLOW=50
   TREND_EMA_FAST=9
   TREND_EMA_SLOW=21
   MIN_SENTIMENT_CONFIDENCE=0.55
   MIN_HOLD_MINUTES=15
   # Optional: risk controls (defaults shown)
   TIME_FILTER_ENABLED=True
   MARKET_OPEN_TIME=09:30
   OPENING_BLACKOUT_MINUTES=30
   POSITION_SIZE_PCT=0.03
   MAX_POSITION_NOTIONAL=2500
   MAX_PORTFOLIO_RISK_PCT=0.01
   MAX_SHARES_PER_TRADE=100
   MAX_CONCURRENT_POSITIONS=2
   STOP_LOSS_MIN_PCT=0.010
   STOP_LOSS_MAX_PCT=0.015
   STOP_LOSS_PCT=0.985
   TAKE_PROFIT_PCT=1.04
   MIN_REWARD_RISK_RATIO=1.0
   TRAILING_STOP_ENABLED=True
   BREAKEVEN_TRIGGER_PCT=0.005
   BREAKEVEN_BUFFER_PCT=0.001
   TRAILING_TRIGGER_PCT=0.010
   TRAILING_STOP_PCT=0.008
   DAILY_LOSS_LIMIT_ENABLED=True
   DAILY_LOSS_LIMIT_USD=250
   DAILY_LOSS_LIMIT_PCT=0.0025
   RISK_STATE_FILE=logs/risk_state.json
   BRACKET_FALLBACK_TO_MARKET=False
   ORDER_REGISTRATION_TIMEOUT=15
   SESSION_START_MESSAGE_ENABLED=True
   ```

   **Strategy and risk controls in detail**
   - `TIME_FILTER_ENABLED` / `MARKET_OPEN_TIME` / `OPENING_BLACKOUT_MINUTES`: no new entry is submitted during the first 30 minutes after the market open. Protective exits, trailing stops and the circuit breaker stay active the whole time. Times use `MARKET_TIMEZONE`, so EST and EDT are handled automatically.
   - `STOP_LOSS_MIN_PCT` / `STOP_LOSS_MAX_PCT`: hard band for every entry's stop loss (default 1.0%-1.5%). `STOP_LOSS_PCT` is clamped into this band, and the final stop is re-validated after whole-cent rounding. Entries that cannot express a valid stop (e.g. very cheap shares) are rejected.
   - `TRAILING_STOP_ENABLED` / `BREAKEVEN_TRIGGER_PCT` / `BREAKEVEN_BUFFER_PCT`: once a trade has reached a 0.5% gain, the stop moves to entry +0.1% so a winner cannot turn into a loser.
   - `TRAILING_TRIGGER_PCT` / `TRAILING_STOP_PCT`: once a trade has reached a 1.0% gain, the stop trails 0.8% behind the highest price seen (the trigger uses the high-water mark, so a pullback does not disarm it). Stops only move up. If the broker rejects a stop update, the bot enforces the same level as a software stop and closes the position at market when it is hit.
   - `MAX_PORTFOLIO_RISK_PCT`: maximum equity at risk per trade when the stop loss is hit (default 1%).
   - `POSITION_SIZE_PCT` / `MAX_POSITION_NOTIONAL`: maximum notional value of one entry (default 3% of available cash, hard-capped at $2,500).
   - `MAX_CONCURRENT_POSITIONS`: maximum number of open positions (default 2). Pending entry orders count against the limit, so a third position cannot be opened accidentally.
   - `MAX_SHARES_PER_TRADE`: hard share cap per entry; set to `0` to disable.
   - `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT`: bracket legs, validated before every entry (stop must be below entry, take profit above it, and the reward/risk must meet `MIN_REWARD_RISK_RATIO`).
   - `INDICATOR_TIMEFRAME` / `INDICATOR_LOOKBACK` / `MIN_BARS_REQUIRED`: indicators run on 5-minute bars (200 bars of history, at least 60 valid bars before a signal is trusted). A symbol with insufficient history is skipped instead of producing a weak signal.
   - `TREND_SMA_FAST` / `TREND_SMA_SLOW` / `TREND_EMA_FAST` / `TREND_EMA_SLOW`: entry requires both a bullish SMA stack and a bullish EMA stack, plus a bullish MACD.
   - `RSI_OVERSOLD` / `MIN_SENTIMENT_CONFIDENCE`: the RSI leg must be at or below this level and FinBERT must return a positive label with at least this confidence. All legs must agree; there is no scoring or partial approval.
   - `MIN_HOLD_MINUTES`: a position cannot be closed by an RSI/MA signal before this many minutes have elapsed. Broker-side stop-loss/take-profit fills, trailing-stop exits and the circuit breaker are exempt and act immediately.
   - `DAILY_LOSS_LIMIT_ENABLED` / `DAILY_LOSS_LIMIT_USD` / `DAILY_LOSS_LIMIT_PCT`: daily circuit breaker. The stricter of -$250 and -0.25% of the day's opening equity is used. When hit, the bot cancels all orders, liquidates every position, sends the Arabic Telegram alert and refuses new entries for the rest of the day. The halt is persisted in `RISK_STATE_FILE`, so restarting the bot does not bypass it; an unreadable state file also fails closed and halts the session. If account data cannot be read, new entries are blocked until the daily loss can be verified again.
   - `BRACKET_FALLBACK_TO_MARKET` / `ORDER_REGISTRATION_TIMEOUT`: every entry is submitted as a bracket with both a stop loss and take profit. The bot waits for the broker to acknowledge the order; a rejected or unacknowledged bracket is reported as a failure (with an alert) and never as a successful BUY. By default the trade is skipped if the protective legs are rejected; set `BRACKET_FALLBACK_TO_MARKET=True` only if you explicitly accept an unprotected market entry.
   - `SESSION_START_MESSAGE_ENABLED`: sends the “New Trading Session” welcome to Telegram on startup and on the first trading iteration of each new day, with the market-local time and current account balance. It is sent only once per startup/session.
   - Order tracking: open stop-loss/take-profit ids are refreshed from Alpaca once per iteration. Closed ids are purged from both the tracker and lumibot's internal order lists, and a replaced stop is tracked under its new id. Automated bracket cancellation produces a debug log entry, not a Telegram alert; genuine user or strategy cancellations still send `تم إلغاء الأمر`.

---

## 🚀 Usage

Run the trading bot live/paper session:
```powershell
python tradingbot.py
```

Run the risk-control test suite:
```powershell
python -m pytest tests -q
```