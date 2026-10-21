"""Alpaca data source pinned to the IEX feed.

lumibot's bundled :class:`~lumibot.data_sources.AlpacaData` (3.0.6) builds every
alpaca-py request without a ``feed`` argument.  alpaca-py therefore falls back
to its default, SIP, and free/paper accounts reject recent data with::

    {"message":"subscription does not permit querying recent SIP data"}

The ``FEED`` key in :func:`config.alpaca_config` is *not* read by lumibot, so
the only reliable fix is to subclass ``AlpacaData`` and explicitly pass
``feed=DataFeed.IEX`` on every stock request.  :class:`AlpacaIEXData` does that
for historical bars and the latest-trade lookup.

The subclass also repairs lumibot's broken multi-minute timestep mapping
(``"5 minutes"`` currently resolves to the invalid ``["51Min", "minute"]``)
and makes the bar lookback generic, so the strategy can request 5-minute bars
without touching lumibot internals.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

import pandas as pd  # pyright: ignore[reportMissingModuleSource]
from alpaca.data.enums import DataFeed  # pyright: ignore[reportMissingImports]
from alpaca.data.historical import (  # pyright: ignore[reportMissingImports]
    CryptoHistoricalDataClient,
    StockHistoricalDataClient,
)
from alpaca.data.requests import (  # pyright: ignore[reportMissingImports]
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
    StockBarsRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # pyright: ignore[reportMissingImports]
from lumibot.data_sources import AlpacaData  # pyright: ignore[reportMissingImports]
from lumibot.entities import Asset  # pyright: ignore[reportMissingImports]

from config import ALPACA_DATA_FEED

logger = logging.getLogger(__name__)

# lumibot 3.0.6 builds these timestep representations incorrectly.
_MULTI_MINUTE_TIMEFRAMES = {
    "5 minutes": 5,
    "10 minutes": 10,
    "15 minutes": 15,
    "30 minutes": 30,
}

_MINUTES_PER_UNIT = {
    TimeFrameUnit.Minute: 1,
    TimeFrameUnit.Hour: 60,
    TimeFrameUnit.Day: 1440,
    TimeFrameUnit.Week: 7 * 1440,
    TimeFrameUnit.Month: 30 * 1440,
}

_MAX_BAR_REQUESTS = 16  # maximum bar requests per call


def resolve_feed(value=ALPACA_DATA_FEED) -> DataFeed:
    """Map an ``ALPACA_DATA_FEED`` string to a :class:`DataFeed` member.

    Unknown values fall back to ``DataFeed.IEX`` so we never accidentally
    request the subscription-gated SIP feed.
    """
    if isinstance(value, DataFeed):
        return value
    try:
        return DataFeed(str(value).strip().lower())
    except ValueError:
        logger.warning("Unknown ALPACA_DATA_FEED=%r; falling back to IEX", value)
        return DataFeed.IEX


def _timeframe_minutes(freq) -> int:
    """Return the duration of one bar in minutes."""
    if isinstance(freq, TimeFrame):
        amount = max(1, int(getattr(freq, "amount", 1)))
        return amount * _MINUTES_PER_UNIT.get(getattr(freq, "unit", None), 1)

    match = re.match(r"\s*(\d+)\s*([A-Za-z]+)", str(freq))
    if not match:
        return 1
    amount = max(1, int(match.group(1)))
    unit = match.group(2).lower().rstrip("s")
    return amount * {
        "min": 1,
        "minute": 1,
        "hour": 60,
        "day": 1440,
        "week": 7 * 1440,
        "month": 30 * 1440,
    }.get(unit, 1)


class AlpacaIEXData(AlpacaData):
    """``AlpacaData`` that always requests the configured (default IEX) feed."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.feed = resolve_feed()

    def _parse_source_timestep(self, timestep, reverse=False):
        """Fix lumibot's multi-minute mapping before delegating."""
        amount = _MULTI_MINUTE_TIMEFRAMES.get(timestep)
        if amount is not None:
            if reverse:
                return TimeFrame(amount, TimeFrameUnit.Minute)
            return timestep
        return super()._parse_source_timestep(timestep, reverse=reverse)

    def get_last_price(self, asset, quote=None, exchange=None, **kwargs):
        """Return the latest trade price, requesting the IEX feed for stocks."""
        if quote is not None:
            if type(asset) == Asset and asset.asset_type == "stock":
                symbol = asset.symbol
            elif isinstance(asset, tuple):
                symbol = f"{asset[0].symbol}/{quote.symbol}"
            else:
                symbol = f"{asset.symbol}/{quote.symbol}"
        elif isinstance(asset, tuple):
            symbol = f"{asset[0].symbol}/{asset[1].symbol}"
        else:
            symbol = asset.symbol

        is_crypto = (
            isinstance(asset, tuple) and asset[0].asset_type == "crypto"
        ) or (isinstance(asset, Asset) and asset.asset_type == "crypto")

        if is_crypto:
            client = CryptoHistoricalDataClient()
            params = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            latest = client.get_crypto_latest_quote(params)
            latest = latest[list(latest.keys())[0]]
            return (latest.bid_price + latest.ask_price) / 2

        client = StockHistoricalDataClient(self.api_key, self.api_secret)
        params = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=self.feed)
        return client.get_stock_latest_trade(params)[symbol].price

    def get_barset_from_api(
        self, asset, freq, limit=None, end=None, start=None, quote=None
    ):
        """Fetch historical bars, passing the IEX ``feed`` to alpaca-py.

        This mirrors ``AlpacaData.get_barset_from_api`` but adds
        ``feed=self.feed`` to the ``StockBarsRequest`` and derives the lookback
        window from the actual bar duration, so 5-minute bars work correctly.
        """
        if isinstance(asset, tuple):
            if quote is None:
                quote = asset[1]
            asset = asset[0]

        limit = limit or 1000
        bar_minutes = _timeframe_minutes(freq)

        if not end:
            # Alpaca limitation of not getting the most recent 15 minutes.
            end = datetime.now(timezone.utc) - timedelta(minutes=15)

        # Extra room for weekends/Mondays when no bars exist, plus a wider
        # window for daily bars (only ~5 bars per calendar week).
        window_minutes = limit * bar_minutes
        if bar_minutes >= 1440:
            window_minutes = int(limit * 1.5) * bar_minutes
        if datetime.now().weekday() == 0:
            window_minutes += 4896

        df = pd.DataFrame()
        attempts = 0
        while len(df) < limit and attempts < _MAX_BAR_REQUESTS:
            attempts += 1
            window_start = start or (end - timedelta(minutes=window_minutes))

            if asset.asset_type == "crypto":
                symbol = f"{asset.symbol}/{quote.symbol}"
                client = CryptoHistoricalDataClient()
                params = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=freq,
                    start=window_start,
                    end=end,
                )
                barset = client.get_crypto_bars(params)
            else:
                symbol = asset.symbol
                client = StockHistoricalDataClient(self.api_key, self.api_secret)
                params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=freq,
                    start=window_start,
                    end=end,
                    feed=self.feed,
                )
                try:
                    barset = client.get_stock_bars(params)
                except Exception as exc:
                    logger.error(
                        "Could not get pricing data from Alpaca for %s: %s",
                        symbol,
                        exc,
                    )
                    return None

            df = getattr(barset, "df", pd.DataFrame())
            if df is None or df.empty:
                logger.error(
                    "Could not get any pricing data from Alpaca for %s; "
                    "the DataFrame came back empty",
                    symbol,
                )
                return None

            # Alpaca returns a MultiIndex; keep only the timestamp index.
            df = df.reset_index(level=0, drop=True)
            df = df[~df.index.duplicated(keep="first")]
            df = df.iloc[-limit:]
            df = df[df.close > 0]

            if start is not None:
                break  # caller asked for a fixed range; do not widen it
            window_minutes *= 2

        if len(df) < limit:
            logger.warning(
                "Dataframe for %s has %d rows while %d were requested; "
                "further data does not exist for Alpaca",
                symbol,
                len(df),
                limit,
            )

        return df
