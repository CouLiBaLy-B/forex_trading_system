"""Pydantic models and exceptions for the execution engine.

All DTOs are Pydantic ``BaseModel`` subclasses providing validation,
serialization, and clear contracts between layers.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_serializer


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderType(str, Enum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(str, Enum):
    """Order direction."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order lifecycle states."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class FillType(str, Enum):
    """Source of a fill."""

    MANUAL = "manual"
    STOP_TRIGGER = "stop_trigger"
    TP_TRIGGER = "tp_trigger"
    AUTO_CANCEL = "auto_cancel"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OrderValidationError(Exception):
    """Raised when an order fails validation checks."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Order validation failed: {reason}")


class MarginExceededError(Exception):
    """Raised when margin requirements cannot be met."""

    def __init__(self, required: float, available: float) -> None:
        self.required = required
        self.available = available
        super().__init__(
            f"Margin exceeded: required {required:.2f} > available {available:.2f}"
        )


class OrderNotFound(Exception):
    """Raised when a referenced order does not exist."""

    def __init__(self, order_id: str) -> None:
        self.order_id = order_id
        super().__init__(f"Order not found: {order_id}")


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

class Quote(BaseModel):
    """Market quote for a single trading pair.

    Attributes:
        symbol: Trading pair (e.g. ``"EURUSD"``).
        price: Mid-market price.
        bid: Bid price.
        ask: Ask price.
        timestamp: Quote timestamp (UTC).
        volume: Reported trading volume (optional).
    """

    symbol: str
    price: float
    bid: float
    ask: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    volume: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    @field_serializer("timestamp")
    def _serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()

    @property
    def spread_pips(self) -> float:
        """Spread in pips."""
        return self.ask - self.bid


class Order(BaseModel):
    """Represents a client order in the execution engine.

    Attributes:
        order_id: Unique identifier (auto-generated UUID).
        symbol: Trading pair (e.g. ``"EURUSD"``).
        order_type: Type of the order.
        side: Buy or sell.
        quantity: Number of lots.
        price: Limit / stop price (``None`` for market orders).
        status: Current lifecycle state.
        fill_price: Filled price (set on fill).
        filled_quantity: Cumulative filled quantity.
        remaining_quantity: Quantity still outstanding.
        stop_loss: Attached stop-loss price (optional).
        take_profit: Attached take-profit price (optional).
        created_at: Order creation timestamp (UTC).
        updated_at: Last status-change timestamp (UTC).
        filled_at: Fill timestamp (``None`` until filled).
        cancelled_at: Cancellation timestamp (``None`` until cancelled).
        rejection_reason: Reason set on rejection (``None`` if not rejected).
        commission: Commission charged on fill (``0`` before fill).
    """

    order_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float = Field(gt=0)
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    commission: float = 0.0

    def model_post_init(self, __context: object) -> None:
        """Set remaining quantity from initial quantity."""
        self.remaining_quantity = self.quantity

    @field_serializer("created_at", "updated_at", "filled_at", "cancelled_at")
    def _serialize_datetimes(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.isoformat()


class FillRecord(BaseModel):
    """A fill event produced by the trading engine.

    Attributes:
        fill_id: Unique fill identifier.
        order_id: Parent order.
        symbol: Trading pair.
        side: Fill direction.
        fill_price: Execution price (including spread/slippage).
        quantity: Filled lots.
        spread_cost: Portion of spread allocated to this fill.
        commission: Commission charged.
        total_cost: spread_cost + commission.
        fill_type: Source of the fill.
        timestamp: Fill timestamp (UTC).
    """

    fill_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    order_id: str
    symbol: str
    side: OrderSide
    fill_price: float
    quantity: float
    spread_cost: float = 0.0
    commission: float = 0.0
    total_cost: float = 0.0
    fill_type: FillType = FillType.MANUAL
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"arbitrary_types_allowed": True}

    @field_serializer("timestamp")
    def _serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()


class OrderBook(BaseModel):
    """In-memory order book tracking active orders.

    Attributes:
        orders: Dictionary of ``order_id -> Order`` for O(1) lookups.
        max_capacity: Soft limit for orders before emitting a warning.
    """

    orders: dict[str, Order] = Field(default_factory=dict)
    max_capacity: int = 10_000

    @property
    def total_quantity(self) -> float:
        """Sum of remaining quantities across all active orders."""
        return sum(
            o.remaining_quantity
            for o in self.orders.values()
            if o.status in (OrderStatus.OPEN, OrderStatus.PENDING)
        )

    @property
    def active_count(self) -> int:
        """Number of open or pending orders."""
        return sum(
            1
            for o in self.orders.values()
            if o.status in (OrderStatus.OPEN, OrderStatus.PENDING)
        )

    def add(self, order: Order) -> None:
        """Insert or update an order in the book.

        Args:
            order: Order to store.
        """
        self.orders[order.order_id] = order

    def remove(self, order_id: str) -> None:
        """Remove an order by ID.

        Args:
            order_id: Identifier of the order to remove.

        Raises:
            OrderNotFound: If the order does not exist.
        """
        if order_id not in self.orders:
            raise OrderNotFound(order_id)
        del self.orders[order_id]

    def get(self, order_id: str) -> Optional[Order]:
        """Retrieve an order by ID (``None`` if absent)."""
        return self.orders.get(order_id)

    def get_active(self, symbol: Optional[str] = None) -> list[Order]:
        """Return all open/pending orders, optionally filtered by symbol."""
        active_statuses = {OrderStatus.OPEN, OrderStatus.PENDING}
        return [
            o
            for o in self.orders.values()
            if o.status in active_statuses
            and (symbol is None or o.symbol == symbol)
        ]


class TradeRecord(BaseModel):
    """Post-fill trade summary.

    Attributes:
        trade_id: Unique trade identifier.
        order_id: Parent order ID.
        fill_id: Associated fill ID.
        symbol: Trading pair.
        side: Fill direction.
        entry_price: Fill price.
        quantity: Lots filled.
        notional: ``abs(entry_price * quantity)``.
        spread_cost: Spread cost portion.
        commission: Commission portion.
        total_cost: spread_cost + commission.
        unrealized_pnl: Current unrealized P&L (updated by engine).
        pnl: Realized P&L (set on close).
        margin_used: Margin reserved for this trade.
        margin_available: Margin still available after reservation.
        entry_time: When the trade was opened.
        close_time: When the trade was closed (``None`` while open).
    """

    trade_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    order_id: str
    fill_id: str
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    notional: float = 0.0
    spread_cost: float = 0.0
    commission: float = 0.0
    total_cost: float = 0.0
    unrealized_pnl: float = 0.0
    pnl: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    entry_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    close_time: Optional[datetime] = None

    model_config = {"arbitrary_types_allowed": True}

    @field_serializer("entry_time", "close_time")
    def _serialize_datetimes(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.isoformat()


class CommissionConfig(BaseModel):
    """Commission scheme for paper trading.

    Attributes:
        fixed_per_lot: Flat fee per lot traded.
        percentage: Percentage of notional value (e.g. 0.001 = 0.1 %).
    """

    fixed_per_lot: float = Field(default=0.0, ge=0)
    percentage: float = Field(default=0.0, ge=0, le=1)

    def calculate(self, quantity: float, price: float) -> float:
        """Compute total commission for a fill.

        Args:
            quantity: Lots filled.
            price: Fill price.

        Returns:
            Commission amount.
        """
        notional = abs(price * quantity)
        return self.fixed_per_lot * quantity + self.percentage * notional


class SpreadConfig(BaseModel):
    """Configurable bid/ask spread for simulation.

    Attributes:
        default_spread_pips: Base spread in pips.
        random_jitter_pips: Random jitter range (uniform) applied on top.
    """

    default_spread_pips: float = Field(default=0.5, ge=0)
    random_jitter_pips: float = Field(default=0.0, ge=0)

    model_config = {"arbitrary_types_allowed": True}

    def apply_spread(self, mid_price: float, side: OrderSide) -> float:
        """Return the simulated execution price including spread slippage.

        For a buy the ask price is used (mid + half spread).
        For a sell the bid price is used (mid - half spread).

        Args:
            mid_price: Mid-market price.
            side: Order direction.

        Returns:
            Simulated execution price.
        """
        import random

        jitter = random.uniform(-self.random_jitter_pips, self.random_jitter_pips)
        half_spread = (self.default_spread_pips + jitter) / 2.0

        if side == OrderSide.BUY:
            return mid_price + half_spread
        return mid_price - half_spread


class MarginState(BaseModel):
    """Current margin tracking for the trading account.

    Attributes:
        initial_margin: Starting equity.
        used_margin: Margin currently reserved by open trades.
        available_margin: Margin still available for new positions.
        total_equity: Current total equity (used + available).
    """

    initial_margin: float
    used_margin: float = 0.0
    available_margin: float = 0.0
    total_equity: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, initial_margin: float, margin_ratio: float = 0.02, **data: object) -> None:
        """Initialise margin state.

        Args:
            initial_margin: Starting account equity.
            margin_ratio: Margin requirement fraction per trade (default 2 %).
        """
        super().__init__(
            initial_margin=initial_margin,
            available_margin=initial_margin,
            total_equity=initial_margin,
            **data,
        )
        self._margin_ratio = margin_ratio

    def reserve(self, notional: float) -> float:
        """Reserve margin for a new position.

        Args:
            notional: Position notional value.

        Returns:
            The amount reserved.

        Raises:
            MarginExceededError: If insufficient margin.
        """
        required = notional * self._margin_ratio  # type: ignore[attr-defined]
        if required > self.available_margin:
            raise MarginExceededError(required, self.available_margin)
        self.used_margin += required
        self.available_margin -= required
        return required

    def release(self, notional: float) -> float:
        """Release margin for a closed position.

        Args:
            notional: Position notional value.

        Returns:
            The amount released.
        """
        released = notional * self._margin_ratio  # type: ignore[attr-defined]
        self.used_margin -= released
        self.available_margin += released
        return released

    def update_equity(self, pnl: float) -> None:
        """Update equity with realized P&L and refresh available margin.

        Args:
            pnl: Realized profit or loss.
        """
        self.total_equity += pnl
        self.available_margin = self.total_equity - self.used_margin
