# 🚀 MLTraderProMax

An asynchronous, production-ready automated algorithmic trading bot built in Python. It integrates **Lumibot** for broker execution (Alpaca), **FinBERT** for real-time financial news sentiment analysis, and multi-threaded processing for lightning-fast performance.

---

## 🛠️ Key Features
- **Concurrent Execution:** Processes the entire asset watchlist in parallel using thread pools.
- **Sentiment Analysis:** Evaluates recent financial headlines via Hugging Face's **FinBERT** transformer model.
- **TTL Caching:** Built-in in-memory caching for news, sentiment scores, and technical indicators to minimize redundant API calls and processing overhead.
- **Resilient Error Handling:** Robust exponential backoff with jitter for network failures and Alpaca rate limits.
- **Automated Alerts:** Dispatches instant status updates (fills, cancellations, buy/sell actions) directly to **Telegram** in clear **Darija/Arabic**, including a once-per-session startup message with the current balance.
- **End-of-Day Report:** When the market closes, logs and sends an Arabic daily summary with daily profit/loss ($ and %), winning vs losing trades, the portfolio balance and open positions.
- **Risk Management:** A hard 1.0%-1.5% stop loss on every entry, break-even/trailing stops that protect profits, a maximum of 2 concurrent positions, a $2,500 (or 3% of cash) cap per trade, an opening-volatility time filter (no entries 09:30–10:00 ET), and a daily -$250 / -0.25% circuit breaker that liquidates the account and halts trading for the day.

---

## 📂 Project Structure
```text
MLTradingBot/
│
├── tradingbot.py         # Main entry point and strategy loop
├── config.py             # Centralised environment and strategy configuration
├── trade_ledger.py       # Persistent FIFO ledger for daily win/loss reporting
├── finbert_utils.py      # FinBERT NLP sentiment pipeline and batch inference
├── cache_utils.py        # Thread-safe in-memory TTL cache implementation
├── logging_setup.py      # Size-bounded rotating file and console logger
├── retry_utils.py        # Exponential backoff and jitter retry decorator
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
   REQUIRE_PROTECTIVE_ORDERS=True
   ORDER_REGISTRATION_TIMEOUT=15
   SESSION_START_MESSAGE_ENABLED=True
   ```

   **Risk controls in detail**
   - `TIME_FILTER_ENABLED` / `MARKET_OPEN_TIME` / `OPENING_BLACKOUT_MINUTES`: no new entry is submitted during the first 30 minutes after the market open. Protective exits, trailing stops and the circuit breaker stay active the whole time. Times use `MARKET_TIMEZONE`, so EST and EDT are handled automatically.
   - `STOP_LOSS_MIN_PCT` / `STOP_LOSS_MAX_PCT`: hard band for every entry's stop loss (default 1.0%-1.5%). `STOP_LOSS_PCT` is clamped into this band, and the final stop is re-validated after whole-cent rounding. Entries that cannot express a valid stop (e.g. very cheap shares) are rejected.
   - `TRAILING_STOP_ENABLED` / `BREAKEVEN_TRIGGER_PCT` / `BREAKEVEN_BUFFER_PCT`: once a trade has reached a 0.5% gain, the stop moves to entry +0.1% so a winner cannot turn into a loser.
   - `TRAILING_TRIGGER_PCT` / `TRAILING_STOP_PCT`: once a trade has reached a 1.0% gain, the stop trails 0.8% behind the highest price seen (the trigger uses the high-water mark, so a pullback does not disarm it). Stops only move up. If the broker rejects a stop update, the bot enforces the same level as a software stop and closes the position at market when it is hit.
   - `MAX_PORTFOLIO_RISK_PCT`: maximum equity at risk per trade when the stop loss is hit (default 1%).
   - `POSITION_SIZE_PCT` / `MAX_POSITION_NOTIONAL`: maximum notional value of one entry (default 3% of available cash, hard-capped at $2,500).
   - `MAX_CONCURRENT_POSITIONS`: maximum number of open positions (default 2). Pending entry orders count against the limit, so a third position cannot be opened accidentally.
   - `MAX_SHARES_PER_TRADE`: hard share cap per entry; set to `0` to disable.
   - `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT`: bracket legs, validated before every entry (stop must be below entry, take profit above it, and the reward/risk must meet `MIN_REWARD_RISK_RATIO`).
   - `DAILY_LOSS_LIMIT_ENABLED` / `DAILY_LOSS_LIMIT_USD` / `DAILY_LOSS_LIMIT_PCT`: daily circuit breaker. The stricter of -$250 and -0.25% of the day's opening equity is used. When hit, the bot cancels all orders, liquidates every position, sends the Arabic Telegram alert and refuses new entries for the rest of the day. The halt is persisted in `RISK_STATE_FILE`, so restarting the bot does not bypass it; an unreadable state file also fails closed and halts the session. If account data cannot be read, new entries are blocked until the daily loss can be verified again.
   - `REQUIRE_PROTECTIVE_ORDERS` / `ORDER_REGISTRATION_TIMEOUT`: every entry is submitted as a bracket with both a stop loss and take profit. The bot waits for the broker to acknowledge the order; a rejected or unacknowledged bracket is reported as a failure (with an alert) and never as a successful BUY. If the broker rejects the protective legs, the trade is skipped — the bot never falls back to an unprotected market order.
   - `SESSION_START_MESSAGE_ENABLED`: sends the “New Trading Session” welcome to Telegram on startup and on the first trading iteration of each new day, with the market-local time and current account balance. It is sent only once per startup/session.

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