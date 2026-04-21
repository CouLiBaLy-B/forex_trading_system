"""Real-time streaming (polling-based) market data consumers.

Because Yahoo Finance does not offer a true WebSocket endpoint, the
``YahooFinanceStreamer`` implements a lightweight polling loop that
pushes fresh ticks to registered callbacks at a configurable interval.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import ClassVar, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Callback type aliases                                                #
# ------------------------------------------------------------------ #

TickCallback = Callable[[pd.DataFrame], Awaitable[None]] | Callable[[pd.DataFrame], None]
ErrorCallback = Callable[[Exception], Awaitable[None]] | Callable[[Exception], None]


# ------------------------------------------------------------------ #
#  Exceptions                                                         #
# ------------------------------------------------------------------ #

class StreamerError(Exception):
    """Base exception for streaming errors."""


class StreamerStoppedError(StreamerError):
    """Raised when a stopped streamer receives data."""


# ------------------------------------------------------------------ #
#  BaseStreamer (ABC)                                                 #
# ------------------------------------------------------------------ #

class BaseStreamer(ABC):
    """Abstract base class for all market-data streamers.

    Streamers push data to registered callbacks instead of pulling on
    demand.  The ``start`` / ``stop`` lifecycle is managed by the
    subclass implementation.
    """

    _POLL_INTERVAL: ClassVar[float] = 1.0  # seconds
    _MAX_RETRIES: ClassVar[int] = 3

    def __init__(self, poll_interval: float | None = None) -> None:
        self._poll_interval = poll_interval or self._POLL_INTERVAL
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._tick_callbacks: list[TickCallback] = []
        self._error_callbacks: list[ErrorCallback] = []

    # -- public lifecycle ------------------------------------------------

    async def start(self) -> None:
        """Start the stream loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._stream_loop())
        logger.info("Streamer %s started (interval=%.1fs)", self.__class__.__name__, self._poll_interval)

    async def stop(self) -> None:
        """Stop the stream loop gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Streamer %s stopped", self.__class__.__name__)

    @property
    def is_running(self) -> bool:
        return self._running

    # -- callback registration -------------------------------------------

    def on_tick(self, callback: TickCallback) -> None:
        """Register a tick callback.

        The callback receives a ``pd.DataFrame`` with tick data.
        """
        self._tick_callbacks.append(callback)

    def on_error(self, callback: ErrorCallback) -> None:
        """Register an error callback."""
        self._error_callbacks.append(callback)

    def remove_on_tick(self, callback: TickCallback) -> None:
        if callback in self._tick_callbacks:
            self._tick_callbacks.remove(callback)

    def remove_on_error(self, callback: ErrorCallback) -> None:
        if callback in self._error_callbacks:
            self._error_callbacks.remove(callback)

    # -- abstract hooks ------------------------------------------------

    @abstractmethod
    async def _fetch_ticks(self) -> pd.DataFrame:
        """Fetch a batch of tick data from the upstream source."""

    @abstractmethod
    async def _on_connect(self) -> None:
        """Perform any one-time setup before the stream loop."""

    # -- internal --------------------------------------------------------

    async def _stream_loop(self) -> None:
        """Main polling loop."""
        await self._on_connect()

        while self._running:
            try:
                df = await self._fetch_ticks()
                if not df.empty:
                    await self._push_ticks(df)
            except asyncio.CancelledError:
                raise
            except StreamerError as exc:
                logger.error("Streamer error: %s", exc)
                await self._notify_errors(exc)
            except Exception:
                logger.exception("Unexpected error in stream loop")
                await self._notify_errors(StreamerError("Unexpected error"))

            await asyncio.sleep(self._poll_interval)

    async def _push_ticks(self, df: pd.DataFrame) -> None:
        """Dispatch ticks to all registered callbacks."""
        for cb in self._tick_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(df)
                else:
                    cb(df)
            except Exception:
                logger.exception("Error in tick callback for %s", self.__class__.__name__)

    async def _notify_errors(self, exc: Exception) -> None:
        for cb in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(exc)
                else:
                    cb(exc)
            except Exception:
                logger.exception("Error in error callback")


# ------------------------------------------------------------------ #
#  YahooFinanceStreamer                                               #
# ------------------------------------------------------------------ #

class YahooFinanceStreamer(BaseStreamer):
    """Polling-based streamer that wraps ``yfinance``.

    Unlike a true WebSocket, this streamer fetches the latest price
    snapshot at a fixed interval and pushes it through registered
    callbacks.
    """

    _POLL_INTERVAL: ClassVar[float] = 5.0  # 5-second default poll

    def __init__(self, poll_interval: float | None = None) -> None:
        super().__init__(poll_interval=poll_interval)
        self._latest_data: dict[str, pd.DataFrame] = {}

    async def _on_connect(self) -> None:
        logger.info("YahooFinanceStreamer connected (polling mode)")

    async def _fetch_ticks(self) -> pd.DataFrame:
        import yfinance as yf  # runtime import to keep optional

        loop = asyncio.get_running_loop()

        def _snapshot() -> pd.DataFrame:
            ticker = yf.Ticker("EURUSD=X")
            info = ticker.fast_info
            if not info:
                raise StreamerError("No fast_info for EURUSD=X")
            return pd.DataFrame([{
                "symbol": "EURUSD=X",
                "price": float(info.last_price),
                "bid": float(info.bid),
                "ask": float(info.ask),
                "volume": float(info.volume),
                "timestamp": dt.datetime.now(dt.timezone.utc),
            }])

        df = await asyncio.wait_for(
            loop.run_in_executor(None, _snapshot),
            timeout=10.0,
        )
        self._latest_data["EURUSD=X"] = df
        return df

    async def get_latest(self, symbol: str = "EURUSD=X") -> pd.DataFrame | None:
        """Return the most recently fetched tick snapshot for *symbol*."""
        return self._latest_data.get(symbol)
