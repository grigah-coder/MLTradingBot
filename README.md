# 🚀 MLTraderProMax

An asynchronous, production-ready automated algorithmic trading bot built in Python. It integrates **Lumibot** for broker execution (Alpaca), **FinBERT** for real-time financial news sentiment analysis, and multi-threaded processing for lightning-fast performance.

---

## 🛠️ Key Features
- **Concurrent Execution:** Processes the entire asset watchlist in parallel using thread pools.
- **Sentiment Analysis:** Evaluates recent financial headlines via Hugging Face's **FinBERT** transformer model.
- **TTL Caching:** Built-in in-memory caching for news, sentiment scores, and technical indicators to minimize redundant API calls and processing overhead.
- **Resilient Error Handling:** Robust exponential backoff with jitter for network failures and Alpaca rate limits.
- **Automated Alerts:** Dispatches instant status updates (fills, cancellations, buy/sell actions) directly to **Telegram** in clear **Darija/Arabic**.
- **End-of-Day Report:** When the market closes, logs and sends an Arabic daily summary with daily profit/loss ($ and %), winning vs losing trades, the portfolio balance and open positions.
- **Risk Management:** Features built-in Stop Loss, Take Profit, and bracket order fallback mechanics.

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
   ```

---

## 🚀 Usage

Run the trading bot live/paper session:
```powershell
python tradingbot.py
```