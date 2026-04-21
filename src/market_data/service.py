"""High-level market data service that orchestrates fetcher, streamer, and cache."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import ClassVar, Optional

import pandas as pd

from .cache import DataCache
from .fetcher import BaseFetcher, FetcherError, FetcherHTTPError, FetcherRateLimitError, FetcherTimeoutError
from .models import Quote
from .streaming import BaseStreamer, TickCallback, StreamerError

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  MarketDataService                                                  #
# ------------------------------------------------------------------ #

class MarketDataService:
    """Orchestrates fetcher, streamer, and cache to provide a unified
    market-data API.

    Public API::

        service = MarketDataService(fetcher, streamer)
        price = await service.get_price("EUR/USD")
        ohlcv = await service.get_ohlcv("EUR/USD", period="1d")
        await service.subscribe("EUR/USD", callback)
        await service.unsubscribe("EUR/USD", callback)

    Args:
        fetcher:  A ``BaseFetcher`` implementation.
        streamer: A ``BaseStreamer`` implementation.
        cache:    Optional shared ``DataCache`` (a new one is created
                  with default TTL of 60 s if omitted).
    """

    _PRICE_TTL: ClassVar[float] = 60.0  # seconds

    def __init__(
        self,
        fetcher: BaseFetcher,
        streamer: BaseStreamer,
        cache: DataCache | None = None,
    ) -> None:
        self.fetcher = fetcher
        self.streamer = streamer
        self._cache = cache or DataCache(default_ttl=self._PRICE_TTL)
        self._subscribers: dict[str, list[TickCallback]] = {}

    # -- price queries -------------------------------------------------

    async def get_price(self, symbol: str) -> Quote:
        """Return the latest price (with bid/ask) for *symbol*.

        Checks the cache first; falls back to the fetcher.
        """
        cache_key = f"price:{symbol}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Price cache HIT for %s", symbol)
            return Quote(**cached)

        logger.info("Fetching live price for %s", symbol)
        df = await self.fetcher.get_quote(symbol)
        if df.empty:
            raise FetcherError(f"No quote data for {symbol}")

        row = df.iloc[0]
        quote = Quote(
            symbol=str(row.get("symbol", symbol)),
            price=float(row["price"]),
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            timestamp=row["timestamp"] if hasattr(row["timestamp"], "tzinfo") else dt.datetime.now(dt.timezone.utc),
            volume=float(row.get("volume", 0.0)),
        )

        self._cache.set(cache_key, quote.model_dump(), ttl=self._PRICE_TTL)
        logger.info("Price updated for %s: %.5f", symbol, quote.price)
        return quote

    async def get_ohlcv(self, symbol: str, period: str = "1d", interval: str = "1d") -> pd.DataFrame:
        """Return OHLCV data for *symbol* (cached by default).

        Args:
            symbol: Trading pair symbol.
            period: Data period string passed to the fetcher.
            interval: Candle interval passed to the fetcher.

        Returns:
            A pandas DataFrame with columns
            ``timestamp, open, high, low, close, volume``.
        """
        cache_key = f"ohlcv:{symbol}:{period}:{interval}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("OHLCV cache HIT for %s (period=%s, interval=%s)", symbol, period, interval)
            return cached

        ohlcv_frame = await self.fetcher.fetch_ohlcv(symbol, period=period, interval=interval)
        df = ohlcv_frame.df

        self._cache.set(cache_key, df.to_dict(orient="list"), ttl=self._PRICE_TTL)
        logger.info("OHLCV updated for %s: %d rows", symbol, len(df))
        return df

    # -- subscription --------------------------------------------------

    async def subscribe(self, symbol: str, callback: TickCallback) -> None:
        """Subscribe to tick updates for *symbol*.

        The callback receives a ``pd.DataFrame`` each time new data
        arrives from the streamer.
        """
        self._subscribers.setdefault(symbol, []).append(callback)
        self.streamer.on_tick(callback)
        logger.info("Subscribed to %s with callback %s", symbol, callback.__name__)

    async def unsubscribe(self, symbol: str, callback: TickCallback) -> None:
        """Unsubscribe *callback* from *symbol*."""
        if symbol in self._subscribers:
            self._subscribers[symbol] = [cb for cb in self._subscribers[symbol] if cb is not callback]
            if not self._subscribers[symbol]:
                del self._subscribers[symbol]
            self.streamer.remove_on_tick(callback)
        logger.info("Unsubscribed from %s", symbol)

    # -- helpers -------------------------------------------------------

    async def get_spread(self, symbol: str) -> float:
        """Return the bid-ask spread in pips for *symbol*."""
        quote = await self.get_price(symbol)
        return (quote.ask - quote.bid) * 10000  # rough pip conversion

    async def health_check(self) -> dict[str, object]:
        """Return service health status."""
        return {
            "cache_size": self._cache.size(),
            "cache_ttl": self._cache.default_ttl,
            "streamer_running": self.streamer.is_running,
            "subscribers": {s: len(cbs) for s, cbs in self._subscribers.items()},
        }
