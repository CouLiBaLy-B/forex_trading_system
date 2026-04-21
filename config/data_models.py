"""Core data models shared across the trading system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class StrategyName(str, Enum):
    MA_CROSSOVER = "MA_CROSSOVER"
    MEAN_REVERSION = "MEAN_REVERSION"
    RSI = "RSI"
    MACD = "MACD"
    BOLLINGER_BANDS = "BOLLINGER_BANDS"


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class OHLCV:
    """Candlestick data point."""
    instrument: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float = 0.0

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        return self.high - self.low


@dataclass(slots=True)
class Tick:
    """Single price tick from market data source."""
    instrument: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int = 0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


# ---------------------------------------------------------------------------
# Order & Execution
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Order:
    """Represents a trade order in the system."""
    instrument: str
    side: Side
    order_type: OrderType
    quantity: float
    price: float | None = None  # for limit orders
    stop_price: float | None = None  # for stop orders
    status: OrderStatus = OrderStatus.PENDING
    id: UUID = field(default_factory=uuid4)
    strategy_name: str = ""
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: datetime | None = None
    rejection_reason: str | None = None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_fully_filled(self) -> bool:
        return self.filled_quantity >= self.quantity

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)


@dataclass(slots=True)
class Fill:
    """Record of an order fill."""
    order_id: UUID
    instrument: str
    side: Side
    quantity: float
    price: float
    commission: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def pnl(self) -> float:
        return 0.0  # depends on position context


# ---------------------------------------------------------------------------
# Position & Portfolio
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Position:
    """Open position in an instrument."""
    instrument: str
    side: Side
    quantity: float
    average_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    opened_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    strategy_name: str = ""
    commission_paid: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.average_entry_price == 0:
            return 0.0
        pnl = self.current_pnl
        return (pnl / (self.quantity * self.average_entry_price)) * 100

    @property
    def current_pnl(self) -> float:
        if self.side == Side.BUY:
            return (self.current_price - self.average_entry_price) * self.quantity
        return (self.average_entry_price - self.current_price) * self.quantity


@dataclass(slots=True)
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_balance: float = 0.0
    cash: float = 0.0
    margin_used: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_equity: float = 0.0
    positions_count: int = 0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    @property
    def equity(self) -> float:
        return self.total_equity


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TradeRecord:
    """Completed trade record."""
    id: UUID = field(default_factory=uuid4)
    instrument: str = ""
    side: Side = Side.BUY
    strategy_name: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: datetime | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    holding_period_seconds: float = 0.0
    exit_reason: str = ""


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SignalEvent:
    """Strategy-generated trading signal."""
    signal_id: UUID = field(default_factory=uuid4)
    strategy_name: str = ""
    instrument: str = ""
    signal: Signal = Signal.HOLD
    strength: float = 0.0  # confidence 0-1
    target_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(slots=True)
class BacktestResult:
    """Complete backtesting results."""
    strategy_name: str = ""
    instrument: str = ""
    start_date: str = ""
    end_date: str = ""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_period: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_commission(self) -> float:
        return 0.0

    @property
    def calmar_ratio(self) -> float:
        if self.max_drawdown_pct == 0:
            return 0.0
        return self.annualized_return / abs(self.max_drawdown_pct)
