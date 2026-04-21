"""Pydantic models and DTOs for the portfolio module.

All data transfer objects are Pydantic BaseModel subclasses providing validation,
serialization, and clear contracts between layers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_serializer


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    """Direction of a trade."""

    BUY = "buy"
    SELL = "sell"


class PositionSide(str, Enum):
    """Current side of an open position."""

    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    """Type of order that created the position."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class TradeStatus(str, Enum):
    """Lifecycle status of a trade."""

    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

class Position(BaseModel):
    """A currently open trading position.

    Attributes:
        position_id: Unique identifier for the position.
        symbol: Trading pair (e.g. ``"EUR/USD"``).
        side: Long or short direction.
        entry_price: Price at which the position was opened.
        quantity: Number of units / lots.
        stop_loss: Current stop-loss price (None if unset).
        take_profit: Current take-profit price (None if unset).
        unrealized_pnl: Current unrealized profit/loss.
        entry_time: Timestamp of position entry.
        strategy: Name of the strategy that opened this position.
        order_type: Type of order used to enter.
    """

    position_id: str
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    entry_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    strategy: str
    order_type: OrderType = OrderType.MARKET

    @field_serializer("entry_time")
    def _serialize_entry_time(self, value: datetime) -> str:
        return value.isoformat()


class PositionSummary(BaseModel):
    """Aggregated snapshot of a position for reporting.

    Attributes:
        position_id: Unique identifier.
        symbol: Trading pair.
        side: Long or short.
        entry_price: Original entry price.
        current_price: Latest market price.
        quantity: Position size.
        unrealized_pnl: Current unrealized P&L.
        unrealized_pnl_pct: Unrealized P&L as a percentage of notional.
        stop_loss: Active stop-loss (None if unset).
        take_profit: Active take-profit (None if unset).
        days_open: Number of full days since entry.
    """

    position_id: str
    symbol: str
    side: str
    entry_price: float
    current_price: float
    quantity: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    days_open: int


class TradeRecord(BaseModel):
    """A completed trade record (closed position).

    Attributes:
        trade_id: Unique identifier for the trade.
        symbol: Trading pair.
        side: Long or short.
        entry_price: Price at entry.
        exit_price: Price at exit.
        quantity: Position size.
        stop_loss_hit: Whether stop-loss triggered.
        take_profit_hit: Whether take-profit triggered.
        realized_pnl: Profit/loss realized on close.
        realized_pnl_pct: Realized P&L as a percentage of notional.
        entry_time: Timestamp of entry.
        exit_time: Timestamp of exit.
        strategy: Strategy name.
        reason: How/why the position was closed.
        max_unrealized_pnl: Peak unrealized P&L during hold.
        max_drawdown: Worst unrealized drawdown during hold.
    """

    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    stop_loss_hit: bool = False
    take_profit_hit: bool = False
    realized_pnl: float
    realized_pnl_pct: float
    entry_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exit_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    strategy: str
    reason: str
    max_unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0

    @field_serializer("entry_time", "exit_time")
    def _serialize_times(self, value: datetime) -> str:
        return value.isoformat()


class PortfolioState(BaseModel):
    """Full snapshot of the portfolio at a point in time.

    Attributes:
        equity: Current account equity.
        cash: Available cash balance.
        margin_used: Margin currently locked by open positions.
        total_realized_pnl: Cumulative realized profit/loss.
        total_unrealized_pnl: Sum of unrealized P&L across open positions.
        open_positions: List of currently open positions.
        closed_positions: List of recently closed positions (most recent first).
        peak_equity: Highest equity value ever reached.
        max_drawdown: Maximum drawdown from peak equity (fraction 0-1).
        trade_count: Total number of closed trades.
    """

    equity: float = 0.0
    cash: float = 0.0
    margin_used: float = 0.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    open_positions: list[Position] = Field(default_factory=list)
    closed_positions: list[TradeRecord] = Field(default_factory=list)
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0

    @property
    def net_equity(self) -> float:
        """Return equity including unrealized P&L."""
        return self.cash + self.total_unrealized_pnl


class PerformanceMetrics(BaseModel):
    """Computed performance statistics for a portfolio.

    Attributes:
        sharpe_ratio: Annualized Sharpe ratio (risk-free rate = 0).
        sortino_ratio: Annualized Sortino ratio (downside deviation).
        max_drawdown: Maximum drawdown from peak equity (fraction 0-1).
        max_drawdown_duration_days: Longest consecutive drawdown period in days.
        win_rate: Ratio of winning trades to total closed trades.
        profit_factor: Gross wins divided by gross losses.
        avg_win: Average profit of winning trades.
        avg_loss: Average loss of losing trades.
        calmar_ratio: Annualized return divided by max drawdown.
        total_return: Total return as a fraction.
        total_trades: Total number of closed trades.
        winning_trades: Number of winning trades.
        losing_trades: Number of losing trades.
        avg_holding_period_days: Average days positions were held.
        total_pnl: Net realized P&L across all trades.
        current_drawdown: Current drawdown from peak equity (fraction 0-1).
    """

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    calmar_ratio: float = 0.0
    total_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_holding_period_days: float = 0.0
    total_pnl: float = 0.0
    current_drawdown: float = 0.0


class RollingMetrics(BaseModel):
    """Performance metrics computed over a rolling window.

    Attributes:
        window_start: Start of the rolling window.
        window_end: End of the rolling window.
        sharpe_ratio: Sharpe ratio within the window.
        max_drawdown: Max drawdown within the window.
        win_rate: Win rate within the window.
        total_pnl: Total P&L within the window.
        total_return: Total return within the window.
        total_trades: Number of closed trades in the window.
    """

    window_start: datetime
    window_end: datetime
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_return: float = 0.0
    total_trades: int = 0

    @field_serializer("window_start", "window_end")
    def _serialize_times(self, value: datetime) -> str:
        return value.isoformat()


class AggregatedPosition(BaseModel):
    """Aggregated position across multiple sub-positions for the same symbol+side.

    Attributes:
        symbol: Trading pair.
        side: Aggregated direction (long or short).
        total_quantity: Sum of all sub-position quantities.
        average_entry_price: Volume-weighted average entry price.
        unrealized_pnl: Combined unrealized P&L.
        stop_loss: Aggregated stop-loss (nearest to market).
        take_profit: Aggregated take-profit (farthest from market).
        position_ids: Original position IDs that were aggregated.
        strategy: Primary strategy name (most frequent).
    """

    symbol: str
    side: str
    total_quantity: float
    average_entry_price: float
    unrealized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_ids: list[str] = Field(default_factory=list)
    strategy: str
