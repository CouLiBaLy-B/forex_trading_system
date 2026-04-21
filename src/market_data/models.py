"""Data models for market data.

All DTOs are Pydantic BaseModel subclasses providing validation,
serialization, and clear contracts between layers.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
#  Pydantic DTOs                                                      #
# ------------------------------------------------------------------ #

class Quote(BaseModel):
    """Current market quote for a single instrument."""

    symbol: str
    price: float
    bid: float
    ask: float
    timestamp: dt.datetime
    volume: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class OHLCV(BaseModel):
    """A single OHLCV candle."""

    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class Tick(BaseModel):
    """A single tick (price update)."""

    symbol: str
    price: float
    volume: float
    timestamp: dt.datetime


class MarketDepthLevel(BaseModel):
    """One level on one side of the order book."""

    price: float
    quantity: float


class MarketDepth(BaseModel):
    """Bid / ask depth snapshot."""

    symbol: str
    bids: list[MarketDepthLevel] = Field(default_factory=list)
    asks: list[MarketDepthLevel] = Field(default_factory=list)
    timestamp: dt.datetime | None = None


# ------------------------------------------------------------------ #
#  DataFrame wrappers                                                 #
# ------------------------------------------------------------------ #

class OHLCVFrame:
    """Wrapper around a pandas DataFrame that carries instrument metadata.

    The inner DataFrame *must* have columns:
        timestamp, open, high, low, close, volume

    Attributes:
        instrument: Trading pair symbol (e.g. ``"EUR/USD"``).
        df: The underlying pandas DataFrame.
    """

    def __init__(self, instrument: str, df: pd.DataFrame) -> None:
        self.instrument = instrument
        self.df = df.reset_index(drop=True)
        self._validate()

    # -- public helpers ------------------------------------------------

    @property
    def latest(self) -> pd.Series:
        """Return the last row as a Series."""
        return self.df.iloc[-1]

    @property
    def closes(self) -> np.ndarray:
        return self.df["close"].to_numpy(dtype=np.float64)

    @property
    def volumes(self) -> np.ndarray:
        return self.df["volume"].to_numpy(dtype=np.float64)

    @property
    def timestamps(self) -> list[dt.datetime]:
        return self.df["timestamp"].dt.to_pydatetime().tolist()  # type: ignore[arg-type]

    def to_ohlcv_list(self) -> list[OHLCV]:
        """Convert to a list of OHLCV Pydantic models."""
        rows: list[OHLCV] = []
        for _, row in self.df.iterrows():  # type: ignore[union-attr]
            rows.append(
                OHLCV(
                    timestamp=row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],  # type: ignore[union-attr]
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
        return rows

    # -- internal ------------------------------------------------------

    def _validate(self) -> None:
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        actual = set(self.df.columns)  # type: ignore[union-attr]
        missing = required - actual
        if missing:
            raise ValueError(f"OHLCVFrame missing columns: {missing}")


class TickFrame:
    """Wrapper around a pandas DataFrame that carries instrument metadata.

    The inner DataFrame *must* have columns:
        symbol, price, volume, timestamp
    """

    def __init__(self, symbol: str, df: pd.DataFrame) -> None:
        self.symbol = symbol
        self.df = df.reset_index(drop=True)
        self._validate()

    @property
    def latest(self) -> pd.Series:
        return self.df.iloc[-1]

    @property
    def prices(self) -> np.ndarray:
        return self.df["price"].to_numpy(dtype=np.float64)

    def to_tick_list(self) -> list[Tick]:
        rows: list[Tick] = []
        for _, row in self.df.iterrows():  # type: ignore[union-attr]
            ts = row["timestamp"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()  # type: ignore[assignment]
            rows.append(
                Tick(symbol=self.symbol, price=float(row["price"]), volume=float(row["volume"]), timestamp=ts)  # type: ignore[union-attr]
            )
        return rows

    def _validate(self) -> None:
        required = {"symbol", "price", "volume", "timestamp"}
        actual = set(self.df.columns)  # type: ignore[union-attr]
        missing = required - actual
        if missing:
            raise ValueError(f"TickFrame missing columns: {missing}")
