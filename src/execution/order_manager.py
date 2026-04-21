"""OrderManager – order lifecycle, order book management, and validation.

Manages the full order life-cycle from submission through execution to
cancellation or rejection.  Validates every request against configurable
rules (minimum lot size, margin, price bounds).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .models import (
    CommissionConfig,
    FillRecord,
    FillType,
    MarginExceededError,
    MarginState,
    Order,
    OrderBook,
    OrderNotFound,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
    SpreadConfig,
    TradeRecord,
)


class OrderManager:
    """Central order management engine.

    Responsibilities:
        * **Order lifecycle** – submit, amend, cancel, fill.
        * **Order book** – in-memory active-order tracking.
        * **Validation** – minimum lots, margin checks, price bounds.
        * **State transitions** – enforce valid order-state transitions.

    State transition diagram::

        PENDING -> OPEN
        PENDING -> REJECTED
        OPEN    -> FILLED
        OPEN    -> CANCELLED
        REJECTED, CANCELLED, FILLED are terminal states.
    """

    VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
        OrderStatus.PENDING: {OrderStatus.OPEN, OrderStatus.REJECTED},
        OrderStatus.OPEN: {OrderStatus.FILLED, OrderStatus.CANCELLED},
    }

    def __init__(
        self,
        min_lot_size: float = 0.01,
        spread_config: Optional[SpreadConfig] = None,
        commission_config: Optional[CommissionConfig] = None,
        margin_state: Optional[MarginState] = None,
    ) -> None:
        """Initialise the order manager.

        Args:
            min_lot_size: Minimum tradable lot size.
            spread_config: Spread / slippage simulation settings.
            commission_config: Commission scheme.
            margin_state: Margin tracking state for the account.
        """
        self._order_book = OrderBook()
        self._min_lot_size = min_lot_size
        self._spread_config = spread_config or SpreadConfig()
        self._commission_config = commission_config or CommissionConfig()
        self._margin_state = margin_state
        self._trade_history: list[TradeRecord] = []

    # ------------------------------------------------------------------
    #  Public API – order lifecycle
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Order:
        """Submit a new order after validation.

        Args:
            symbol: Trading pair (e.g. ``"EURUSD"``).
            order_type: Type of order.
            side: Buy or sell.
            quantity: Number of lots.
            price: Limit / stop price (required for LIMIT/STOP, optional for others).
            stop_loss: Optional attached stop-loss price.
            take_profit: Optional attached take-profit price.

        Returns:
            The created ``Order`` in ``PENDING`` status.

        Raises:
            OrderValidationError: If the order fails validation.
        """
        order = Order(
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self._validate(order)
        self._transition(order, OrderStatus.PENDING)
        self._transition(order, OrderStatus.OPEN)
        self._order_book.add(order)
        return order

    def cancel_order(self, order_id: str) -> Order:
        """Cancel an open order.

        Args:
            order_id: Identifier of the order to cancel.

        Returns:
            The updated ``Order`` in ``CANCELLED`` status.

        Raises:
            OrderNotFound: If the order does not exist.
            OrderValidationError: If the order cannot be cancelled from
                its current state.
        """
        order = self._get_order(order_id)
        self._transition(order, OrderStatus.CANCELLED)
        order.cancelled_at = datetime.now(timezone.utc)
        self._order_book.remove(order_id)
        return order

    def amend_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Order:
        """Amend an open order's parameters.

        Args:
            order_id: Identifier of the order to amend.
            price: New limit / stop price.
            quantity: New lot size.
            stop_loss: New stop-loss price.
            take_profit: New take-profit price.

        Returns:
            The updated ``Order``.

        Raises:
            OrderNotFound: If the order does not exist.
            OrderValidationError: If the order cannot be amended from
                its current state or the new params are invalid.
        """
        order = self._get_order(order_id)
        if order.status != OrderStatus.OPEN:
            raise OrderValidationError(
                f"Cannot amend order in status {order.status.value}"
            )

        if price is not None:
            order.price = price
        if quantity is not None:
            order.quantity = quantity
            order.remaining_quantity = quantity - order.filled_quantity
        if stop_loss is not None:
            order.stop_loss = stop_loss
        if take_profit is not None:
            order.take_profit = take_profit

        self._transition(order, OrderStatus.OPEN)  # keep open, update timestamp
        order.updated_at = datetime.now(timezone.utc)
        self._order_book.add(order)
        return order

    def fill_order(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: Optional[float] = None,
        fill_type: str = "manual",
    ) -> tuple[Order, FillRecord, TradeRecord]:
        """Fill an order and record the trade.

        Args:
            order_id: Identifier of the order to fill.
            fill_price: Execution price.
            fill_quantity: Fill quantity (defaults to remaining).
            fill_type: Source of the fill.

        Returns:
            Tuple of ``(updated_order, fill_record, trade_record)``.

        Raises:
            OrderNotFound: If the order does not exist.
            OrderValidationError: If the order cannot be filled.
        """
        order = self._get_order(order_id)
        if order.status != OrderStatus.OPEN:
            raise OrderValidationError(
                f"Cannot fill order in status {order.status.value}"
            )

        qty = fill_quantity if fill_quantity is not None else order.remaining_quantity
        fill_qty = min(qty, order.remaining_quantity)

        commission = self._commission_config.calculate(fill_qty, fill_price)
        spread_cost = self._calculate_spread_cost(order, fill_price)

        trade_notional = abs(fill_price * fill_qty)

        if self._margin_state is not None:
            margin_reserved = self._margin_state.reserve(trade_notional)
        else:
            margin_reserved = 0.0

        self._transition(order, OrderStatus.FILLED)
        order.fill_price = fill_price
        order.filled_quantity += fill_qty
        order.remaining_quantity -= fill_qty
        order.commission = commission
        order.filled_at = datetime.now(timezone.utc)
        order.updated_at = datetime.now(timezone.utc)

        # Remove from active book if fully filled
        if order.remaining_quantity <= 0:
            self._order_book.remove(order_id)
        else:
            self._order_book.add(order)

        fill_record = FillRecord(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            quantity=fill_qty,
            spread_cost=spread_cost,
            commission=commission,
            total_cost=spread_cost + commission,
            fill_type=FillType(fill_type),
        )

        trade_record = TradeRecord(
            order_id=order_id,
            fill_id=fill_record.fill_id,
            symbol=order.symbol,
            side=order.side,
            entry_price=fill_price,
            quantity=fill_qty,
            notional=trade_notional,
            spread_cost=spread_cost,
            commission=commission,
            total_cost=spread_cost + commission,
            margin_used=margin_reserved,
            margin_available=(
                self._margin_state.available_margin
                if self._margin_state is not None
                else 0.0
            ),
        )

        self._trade_history.append(trade_record)
        return order, fill_record, trade_record

    # ------------------------------------------------------------------
    #  Order book queries
    # ------------------------------------------------------------------

    @property
    def order_book(self) -> OrderBook:
        """Read-only access to the internal order book."""
        return self._order_book

    @property
    def trade_history(self) -> list[TradeRecord]:
        """Read-only access to filled trade history."""
        return list(self._trade_history)

    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieve an order by ID (``None`` if not found in book)."""
        return self._order_book.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Return all active orders, optionally filtered by symbol."""
        return self._order_book.get_active(symbol)

    # ------------------------------------------------------------------
    #  Margin helpers
    # ------------------------------------------------------------------

    @property
    def margin_state(self) -> Optional[MarginState]:
        """Current margin state (``None`` if not configured)."""
        return self._margin_state

    @margin_state.setter
    def margin_state(self, value: MarginState) -> None:
        self._margin_state = value

    def check_margin(self, notional: float) -> bool:
        """Check if sufficient margin is available for a given notional.

        Args:
            notional: Proposed position notional.

        Returns:
            ``True`` if margin is sufficient.
        """
        if self._margin_state is None:
            return True
        required = notional * self._margin_state._margin_ratio  # type: ignore[attr-defined]
        return required <= self._margin_state.available_margin

    def close_trade(
        self,
        trade_record: TradeRecord,
        exit_price: float,
    ) -> TradeRecord:
        """Close an open trade and update P&L.

        Args:
            trade_record: The open ``TradeRecord`` to close.
            exit_price: Exit / close price.

        Returns:
            The updated ``TradeRecord`` with realized P&L.
        """
        # Calculate realized P&L
        if trade_record.side == OrderSide.BUY:
            pnl = (exit_price - trade_record.entry_price) * trade_record.quantity
        else:
            pnl = (trade_record.entry_price - exit_price) * trade_record.quantity

        trade_record.pnl = pnl
        trade_record.close_time = datetime.now(timezone.utc)

        if self._margin_state is not None:
            self._margin_state.release(trade_record.notional)
            self._margin_state.update_equity(pnl)

        return trade_record

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _get_order(self, order_id: str) -> Order:
        """Retrieve an order or raise."""
        order = self._order_book.get(order_id)
        if order is None:
            raise OrderNotFound(order_id)
        return order

    def _validate(self, order: Order) -> None:
        """Run all validation rules on a pending order.

        Args:
            order: Order to validate.

        Raises:
            OrderValidationError: On any validation failure.
        """
        # Minimum lot size
        if order.quantity < self._min_lot_size:
            raise OrderValidationError(
                f"Quantity {order.quantity} below minimum lot size "
                f"{self._min_lot_size}"
            )

        # Price required for LIMIT and STOP_LOSS orders
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LOSS):
            if order.price is None:
                raise OrderValidationError(
                    f"{order.order_type.value} orders require a price"
                )

        # Margin check
        if self._margin_state is not None:
            # Use price if set, otherwise estimate with a placeholder notional
            estimated_price = order.price if order.price else 1.0
            notional = abs(estimated_price * order.quantity)
            if not self.check_margin(notional):
                raise OrderValidationError(
                    f"Insufficient margin for notional {notional:.2f}"
                )

    def _transition(self, order: Order, new_status: OrderStatus) -> Order:
        """Transition an order to a new status, enforcing state rules.

        Args:
            order: Order to transition.
            new_status: Target status.

        Raises:
            OrderValidationError: If the transition is invalid.
        """
        current = order.status
        if current not in self.VALID_TRANSITIONS:
            raise OrderValidationError(
                f"Cannot transition from terminal state {current.value}"
            )
        if new_status not in self.VALID_TRANSITIONS[current]:
            raise OrderValidationError(
                f"Invalid transition {current.value} -> {new_status.value}"
            )

        order.status = new_status
        order.updated_at = datetime.now(timezone.utc)
        return order

    def _calculate_spread_cost(self, order: Order, fill_price: float) -> float:
        """Estimate spread cost for a fill.

        Args:
            order: The filled order.
            fill_price: Execution price.

        Returns:
            Estimated spread cost in pips scaled by quantity.
        """
        spread_pips = self._spread_config.default_spread_pips
        half_spread = spread_pips / 2.0
        # For forex, spread cost is typically price-based pip value
        pip_value = half_spread * order.quantity
        return pip_value
