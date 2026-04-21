"""Portfolio module for forex trading.

Exports:
    * ``PortfolioManager`` – position management, P&L, snapshots.
    * ``PerformanceTracker`` – Sharpe, Sortino, drawdown, win rate, etc.
    * Pydantic DTOs: ``Position``, ``TradeRecord``, ``PortfolioState``,
      ``PerformanceMetrics``, ``PositionSummary``, ``AggregatedPosition``,
      ``RollingMetrics``.
    * Enums: ``OrderSide``, ``PositionSide``, ``OrderType``, ``TradeStatus``.
"""

from .manager import PortfolioManager
from .models import (
    AggregatedPosition,
    OrderSide,
    OrderType,
    PerformanceMetrics,
    PortfolioState,
    Position,
    PositionSide,
    PositionSummary,
    RollingMetrics,
    TradeRecord,
    TradeStatus,
)
from .performance import PerformanceTracker

__all__ = [
    "AggregatedPosition",
    "OrderSide",
    "OrderType",
    "PerformanceMetrics",
    "PerformanceTracker",
    "PortfolioManager",
    "PortfolioState",
    "Position",
    "PositionSide",
    "PositionSummary",
    "RollingMetrics",
    "TradeRecord",
    "TradeStatus",
]
