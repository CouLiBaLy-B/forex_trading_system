"""Market data layer for the forex trading system.

Public API
----------
* ``MarketDataService`` -- high-level orchestrator (fetcher + streamer + cache)
* ``BaseFetcher`` / ``BaseStreamer`` -- abstract base classes
* ``YahooFinanceFetcher`` / ``YahooFinanceStreamer`` -- Yahoo Finance implementations
* ``DataCache`` / ``cached`` -- TTL in-memory cache with SQLite fallback
* ``OHLCVFrame`` / ``TickFrame`` -- DataFrame wrappers
* DTOs: ``Quote``, ``OHLCV``, ``Tick``, ``MarketDepth``
* Exceptions: ``FetcherError``, ``FetcherTimeoutError``, ``FetcherRateLimitError``,
  ``FetcherHTTPError``, ``StreamerError``, ``StreamerStoppedError``, ``CacheMissError``
"""

from .fetcher import (
    BaseFetcher,
    FetcherError,
    FetcherHTTPError,
    FetcherRateLimitError,
    FetcherTimeoutError,
    YahooFinanceFetcher,
)
from .models import (
    OHLCV,
    OHLCVFrame,
    MarketDepth,
    MarketDepthLevel,
    Quote,
    Tick,
    TickFrame,
)
from .service import MarketDataService
from .streaming import BaseStreamer, StreamerError, StreamerStoppedError, YahooFinanceStreamer
from .cache import DataCache, CacheError, CacheMissError, cached

__all__ = [
    # service
    "MarketDataService",
    # fetcher
    "BaseFetcher",
    "YahooFinanceFetcher",
    "FetcherError",
    "FetcherTimeoutError",
    "FetcherRateLimitError",
    "FetcherHTTPError",
    # streamer
    "BaseStreamer",
    "YahooFinanceStreamer",
    "StreamerError",
    "StreamerStoppedError",
    # cache
    "DataCache",
    "CacheError",
    "CacheMissError",
    "cached",
    # models
    "OHLCVFrame",
    "TickFrame",
    "Quote",
    "OHLCV",
    "Tick",
    "MarketDepth",
    "MarketDepthLevel",
]
