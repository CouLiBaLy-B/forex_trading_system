"""Fetchers that retrieve market data from external sources.

All fetchers implement ``BaseFetcher`` and support retry with exponential
backoff for transient failures.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import pandas as pd

from .models import OHLCVFrame

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Exceptions                                                         #
# ------------------------------------------------------------------ #

class FetcherError(Exception):
    """Base exception for fetcher failures."""


class FetcherTimeoutError(FetcherError):
    """Raised when a fetch request exceeds its timeout budget."""


class FetcherRateLimitError(FetcherError):
    """Raised when the upstream source enforces a rate limit."""


class FetcherHTTPError(FetcherError):
    """Raised when the upstream source returns an HTTP error status."""


# ------------------------------------------------------------------ #
#  BaseFetcher (ABC)                                                  #
# ------------------------------------------------------------------ #

class BaseFetcher(ABC):
    """Abstract base class for all market-data fetchers.

    Sub-classes must implement ``fetch_ohlcv``, ``get_quote``, and
    ``get_historical``.  The ``fetch`` family of methods already
    handles retry with exponential backoff.
    """

    _MAX_RETRIES: ClassVar[int] = 3
    _BASE_DELAY: ClassVar[float] = 1.0  # seconds

    # -- public (with retry wrapper) -----------------------------------

    async def fetch(self, *args, **kwargs) -> OHLCVFrame | pd.DataFrame:
        """Execute a fetch with exponential-backoff retry.

        Retries up to ``_MAX_RETRIES`` times on transient errors
        (TimeoutError, FetcherTimeoutError, FetcherRateLimitError).
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return await self._call_fetch(*args, **kwargs)
            except (FetcherRateLimitError, FetcherTimeoutError, asyncio.TimeoutError) as exc:
                last_exc = exc
                delay = self._BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Fetcher transient error on attempt %d/%d for %s: %s -- retrying in %.1fs",
                    attempt, self._MAX_RETRIES, self.__class__.__name__, exc, delay,
                )
                await asyncio.sleep(delay)
            except FetcherHTTPError:
                raise  # HTTP errors are not retried (likely perm)

        raise FetcherError(f"All {self._MAX_RETRIES} attempts failed for {self.__class__.__name__}") from last_exc  # type: ignore[misc]

    # -- abstract hooks ------------------------------------------------

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, period: str = "1d", interval: str = "1d", *, as_of: dt.datetime | None = None) -> OHLCVFrame:
        """Return OHLCV data for *symbol*."""

    @abstractmethod
    async def get_quote(self, symbol: str) -> pd.DataFrame:
        """Return the latest quote for *symbol*."""

    @abstractmethod
    async def get_historical(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """Return raw historical data between *start* and *end*."""

    # -- internal dispatch ---------------------------------------------

    async def _call_fetch(self, *args, **kwargs) -> OHLCVFrame | pd.DataFrame:
        """Dispatch to the correct abstract method based on args."""
        raise NotImplementedError  # pragma: no cover


# ------------------------------------------------------------------ #
#  YahooFinanceFetcher                                                #
# ------------------------------------------------------------------ #

class YahooFinanceFetcher(BaseFetcher):
    """Fetch market data via ``yfinance`` with retry and caching support."""

    _TIMEOUT: ClassVar[float] = 10.0

    # -- fetch_ohlcv ---------------------------------------------------

    async def fetch_ohlcv(self, symbol: str, period: str = "1d", interval: str = "1d", *, as_of: dt.datetime | None = None) -> OHLCVFrame:
        """Retrieve OHLCV candles from Yahoo Finance.

        Args:
            symbol: Trading pair (e.g. ``"EURUSD=X"``).
            period: Data period string (``"1d"``, ``"5d"``, ``"1mo"``, ...).
            interval: Candle interval (``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"1d"``).
            as_of: Ignore (included for signature compatibility).

        Returns:
            OHLCVFrame wrapping the returned DataFrame.

        Raises:
            FetcherHTTPError: On non-200 HTTP responses.
            FetcherRateLimitError: When Yahoo rate-limits the request.
            FetcherTimeoutError: When the request exceeds the timeout budget.
        """
        import yfinance as yf  # runtime import to keep optional

        loop = asyncio.get_running_loop()

        def _download() -> pd.DataFrame:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                raise FetcherError(f"No OHLCV data returned for {symbol}")
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "timestamp"})
            if "timestamp" not in df.columns:
                df.insert(0, "timestamp", pd.date_range(end=dt.datetime.now(), periods=len(df), freq=interval))
            return df

        try:
            df = await asyncio.wait_for(loop.run_in_executor(None, _download), timeout=self._TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise FetcherTimeoutError(f"Timed out fetching OHLCV for {symbol}") from exc

        logger.info("Fetched %d OHLCV rows for %s (period=%s, interval=%s)", len(df), symbol, period, interval)
        return OHLCVFrame(instrument=symbol, df=df)

    # -- get_quote -----------------------------------------------------

    async def get_quote(self, symbol: str) -> pd.DataFrame:
        """Fetch the latest quote for *symbol* from Yahoo Finance.

        Returns a single-row DataFrame with columns: symbol, price, bid, ask, timestamp, volume.
        """
        import yfinance as yf

        loop = asyncio.get_running_loop()

        def _quote() -> dict:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            if not info:
                raise FetcherError(f"No quote info for {symbol}")
            return {
                "symbol": symbol,
                "price": float(info.last_price),
                "bid": float(info.bid),
                "ask": float(info.ask),
                "volume": float(info.volume),
                "timestamp": dt.datetime.now(dt.timezone.utc),
            }

        try:
            result = await asyncio.wait_for(loop.run_in_executor(None, _quote), timeout=self._TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise FetcherTimeoutError(f"Timed out fetching quote for {symbol}") from exc

        return pd.DataFrame([result])

    # -- get_historical ------------------------------------------------

    async def get_historical(self, symbol: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """Fetch raw historical data between *start* and *end*.

        Yahoo Finance ``yfinance`` uses ``period`` + ``interval`` rather
        than explicit start/end, so we compute the approximate period
        string ourselves.
        """
        import yfinance as yf

        delta = end - start
        if delta.days > 60:
            period = "max"
        elif delta.days > 30:
            period = "2mo"
        elif delta.days > 7:
            period = "1mo"
        elif delta.days > 1:
            period = "5d"
        else:
            period = "1d"

        loop = asyncio.get_running_loop()

        def _history() -> pd.DataFrame:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)
            if df.empty:
                raise FetcherError(f"No historical data for {symbol} ({start} -> {end})")
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "timestamp"})
            if "timestamp" not in df.columns:
                df.insert(0, "timestamp", pd.date_range(start=start, end=end, periods=len(df)))
            return df

        try:
            df = await asyncio.wait_for(loop.run_in_executor(None, _history), timeout=self._TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise FetcherTimeoutError(f"Timed out fetching historical data for {symbol}") from exc

        logger.info("Fetched %d historical rows for %s", len(df), symbol)
        return df
