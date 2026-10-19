"""Alpaca data source pinned to the IEX feed.

lumibot's bundled :class:`~lumibot.data_sources.AlpacaData` (3.0.6) builds every
alpaca-py request without a ``feed`` argument.  alpaca-py therefore falls back
to its default, SIP, and free/paper accounts reject recent data with::

    {"message":"subscription does not permit querying recent SIP data"}

The ``FEED`` key in :func:`config.alpaca_config` is *not* read by lumibot, so
the only reliable fix is to subclass ``AlpacaData`` and explicitly pass
``feed=DataFeed.IEX`` on every stock request.  :class:`AlpacaIEXData` does that
for both historical bars and the latest trade lookup.
"""

from __future__ import annotations

import logging
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
from lumibot.data_sources import AlpacaData  # pyright: ignore[reportMissingImports]
from lumibot.entities import Asset  # pyright: ignore[reportMissingImports]

from config import ALPACA_DATA_FEED

logger = logging.getLogger(__name__)


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


class AlpacaIEXData(AlpacaData):
    """``AlpacaData`` that always requests the configured (default IEX) feed."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.feed = resolve_feed()

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

        if isinstance(asset, tuple) and asset[0].asset_type == "crypto":
            client = CryptoHistoricalDataClient()
            quote_params = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            latest = client.get_crypto_latest_quote(quote_params)
            latest = latest[list(latest.keys())[0]]
            price = (latest.bid_price + latest.ask_price) / 2
        elif isinstance(asset, Asset) and asset.asset_type == "crypto":
            client = CryptoHistoricalDataClient()
            quote_params = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            latest = client.get_crypto_latest_quote(quote_params)
            latest = latest[list(latest.keys())[0]]
            price = (latest.bid_price + latest.ask_price) / 2
        else:
            client = StockHistoricalDataClient(self.api_key, self.api_secret)
            params = StockLatestTradeRequest(
                symbol_or_symbols=symbol, feed=self.feed
            )
            trade = client.get_stock_latest_trade(params)[symbol]
            price = trade.price

        return price

    def get_barset_from_api(
        self, asset, freq, limit=None, end=None, start=None, quote=None
    ):
        """Fetch historical bars, passing the IEX ``feed`` to alpaca-py.

        This mirrors ``AlpacaData.get_barset_from_api`` but adds
        ``feed=self.feed`` to the ``StockBarsRequest`` so recent SIP data is
        never requested.
        """
        if isinstance(asset, tuple):
            if quote is None:
                quote = asset[1]
            asset = asset[0]

        if not limit:
            limit = 1000

        if not end:
            # Alpaca limitation of not getting the most recent 15 minutes.
            end = datetime.now(timezone.utc) - timedelta(minutes=15)

        loop_limit = limit
        if not start:
            if str(freq) == "1Min":
                if datetime.now().weekday() == 0:  # Monday: prior days were off
                    loop_limit = limit + 4896
                else:
                    loop_limit = limit
            elif str(freq) == "1Day":
                loop_limit = limit * 1.5

        df = pd.DataFrame()

        # Arbitrary limit of 64 iterations before giving up.
        while loop_limit / limit <= 64 and len(df) < limit:
            if str(freq) == "1Min":
                start = end - timedelta(minutes=loop_limit)
            elif str(freq) == "1Day":
                start = end - timedelta(days=loop_limit)

            if asset.asset_type == "crypto":
                symbol = f"{asset.symbol}/{quote.symbol}"
                client = CryptoHistoricalDataClient()
                params = CryptoBarsRequest(
                    symbol_or_symbols=symbol, timeframe=freq, start=start, end=end
                )
                barset = client.get_crypto_bars(params)
            else:
                symbol = asset.symbol
                client = StockHistoricalDataClient(self.api_key, self.api_secret)
                params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=freq,
                    start=start,
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

            df = barset.df

            # Alpaca returns a MultiIndex; keep only the timestamp index.
            df = df.reset_index(level=0, drop=True)

            if df.empty:
                logger.error(
                    "Could not get any pricing data from Alpaca for %s; "
                    "the DataFrame came back empty",
                    symbol,
                )
                return None

            df = df[~df.index.duplicated(keep="first")]
            df = df.iloc[-limit:]
            df = df[df.close > 0]
            loop_limit *= 2

        if len(df) < limit:
            logger.warning(
                "Dataframe for %s has %d rows while %d were requested; "
                "further data does not exist for Alpaca",
                symbol,
                len(df),
                limit,
            )

        return df
