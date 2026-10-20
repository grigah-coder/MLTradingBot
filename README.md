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
- **Risk Management:** Opening-volatility time filter (no trades 09:30–10:00 ET), risk-based position sizing with hard caps, and strictly enforced Stop Loss / Take Profit on every entry via bracket orders.

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
   POSITION_SIZE_PCT=0.02
   MAX_PORTFOLIO_RISK_PCT=0.01
   MAX_SHARES_PER_TRADE=100
   STOP_LOSS_PCT=0.98
   TAKE_PROFIT_PCT=1.04
   MIN_REWARD_RISK_RATIO=1.0
   REQUIRE_PROTECTIVE_ORDERS=True
   SESSION_START_MESSAGE_ENABLED=True
   ```

   **Risk controls in detail**
   - `TIME_FILTER_ENABLED` / `MARKET_OPEN_TIME` / `OPENING_BLACKOUT_MINUTES`: no strategy order (buy or sell) is submitted during the first 30 minutes after the market open. Times use `MARKET_TIMEZONE`, so EST and EDT are handled automatically.
   - `MAX_PORTFOLIO_RISK_PCT`: maximum equity at risk per trade when the stop loss is hit (default 1%).
   - `POSITION_SIZE_PCT`: maximum notional value of one entry as a fraction of available cash (default 2%).
   - `MAX_SHARES_PER_TRADE`: hard share cap per entry; set to `0` to disable.
   - `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT`: bracket legs, validated before every entry (stop must be below entry, take profit above it, and the reward/risk must meet `MIN_REWARD_RISK_RATIO`).
   - `REQUIRE_PROTECTIVE_ORDERS`: when `True` (default) an entry is only placed if the broker accepts both the stop loss and take profit. If the bracket is rejected, the trade is skipped and an alert is sent — the bot never falls back to an unprotected market order.
   - `SESSION_START_MESSAGE_ENABLED`: sends the “New Trading Session” welcome to Telegram on startup and on the first trading iteration of each new day, with the market-local time and current account balance. It is sent only once per startup/session.

---

## 🚀 Usage

Run the trading bot live/paper session:
```powershell
python tradingbot.py
```