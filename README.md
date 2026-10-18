# TraderBot
Build a trader bot which looks at sentiment of live news events and trades appropriately. 

## See it live and in action 📺
<img src="https://i.imgur.com/FaQH8rz.png"/>

# Startup 🚀
1. Create a virtual environment `conda create -n trader python=3.10` 
2. Activate it `conda activate trader`
3. Install initial deps `pip install lumibot timedelta alpaca-trade-api==3.1.1`
4. Install transformers and friends `pip install torch torchvision torchaudio transformers` 
5. Update the `API_KEY` and `API_SECRET` with values from your Alpaca account 
6. Run the bot `python tradingbot.py`

<p>N.B. Torch installation instructions will vary depending on your operating system and hardware. See here for more: 
<a href="pytorch.org/">PyTorch Installation Instructions</a></p>

# Configuration ⚙️

All operational values are read from environment variables (`.env`), so nothing
is hardcoded in the trading logic. Common ones:

| Variable | Default | Purpose |
|---|---|---|
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | – | Alpaca credentials |
| `ALPACA_PAPER` | `true` | Paper vs. live trading |
| `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID` | – | Telegram alerts |
| `TELEGRAM_ENABLED` | `true` | Toggle alerting |
| `WATCHLIST` | `AAPL,NVDA,TSLA,MSFT,AMZN,META,GOOGL` | Symbols (comma-separated) |
| `MAX_WORKERS` | `min(8, len(WATCHLIST))` | Concurrent analyses |
| `SLEEPTIME` | `1M` | Loop interval |
| `RSI_OVERSOLD` / `RSI_OVERBOUGHT` | `40` / `70` | Signal thresholds |
| `POSITION_SIZE_PCT` | `0.05` | Fraction of cash per buy |
| `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT` | `0.98` / `1.04` | Bracket levels |
| `ORDER_TYPE` | `bracket` | Entry order type (`bracket`/`market`) |
| `NEWS_CACHE_TTL` / `SENTIMENT_CACHE_TTL` / `INDICATOR_CACHE_TTL` | `300` / `300` / `30` | Cache TTL (seconds) |
| `RETRY_ATTEMPTS` / `RETRY_BASE_DELAY` / `RETRY_MAX_DELAY` | `4` / `1.0` / `30.0` | Backoff policy |
| `LOG_LEVEL` / `LOG_DIR` / `LOG_FILE` | `INFO` / `logs` / `tradingbot.log` | Logging |
| `IS_BACKTESTING` / `BACKTEST_START` / `BACKTEST_END` | `false` / `2024-01-01` / `2024-01-10` | Backtest mode |

Logs are written to a size-bounded rotating file (`logs/tradingbot.log`) and to
stdout. Outbound calls (Alpaca news/prices/orders, Telegram) are retried with
exponential backoff + jitter and understand Alpaca rate-limit (429) responses.


If you're getting an SSL error when you attempt to call out to the Alpaca Trading api, you'll need to install the required SSL certificates into your machine.
1. Download the following intermediate SSL Certificates, these are required to communicate with Alpaca
* https://letsencrypt.org/certs/lets-encrypt-r3.pem 
* https://letsencrypt.org/certs/isrg-root-x1-cross-signed.pem 
2. Once downloaded, change the file extension of each file to `.cer` 
3. Double click the file and run through the wizard to install it, use all of the default selections. 

</br>
# Other References 🔗

<p>-<a href="github.com/Lumiwealth/lumibot)">Lumibot</a>:trading bot library, makes lifecycle stuff easier .</p>

# Who, When, Why?

👨🏾‍💻 Author: Nick Renotte <br />
📅 Version: 1.x<br />
📜 License: This project is licensed under the MIT License </br>
