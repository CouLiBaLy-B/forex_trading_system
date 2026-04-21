"""PaperTradingEngine – paper / simulated trading with realistic execution.

Simulates fills at current market prices with configurable spread slippage
and a commission model (fixed-per-lot + percentage of notional).  Tracks
margin usage and generates fill / trade records automatically when orders
are triggered (STOP_LOSS / TAKE_PROFIT).
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Optional

from .models import (
    CommissionConfig,
    FillRecord,
    MarginState,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
    SpreadConfig,
    TradeRecord,
)
from .order_manager import OrderManager


class PaperTradingEngine:
    """Paper-trading engine with realistic fill simulation.

    Responsibilities:
        * **Quote management** – current / historical bid-ask quotes.
        * **Fill simulation** – apply spread slippage, generate fills.
        * **Margin tracking** – reserve / release margin for positions.
        * **Trigger management** – evaluate STOP_LOSS / TAKE_PROFIT triggers.
        * **Commission calculation** – fixed + variable model.

    The engine advances time implicitly: call ``advance_quote`` to push a new
    price and trigger any applicable order events.
    """

    def __init__(
        self,
        min_lot_size: float = 0.01,
        spread_config: Optional[SpreadConfig] = None,
        commission_config: Optional[CommissionConfig] = None,
        margin_state: Optional[MarginState] = None,
    ) -> None:
        """Initialise the paper trading engine.

        Args:
            min_lot_size: Minimum tradable lot size.
            spread_config: Spread simulation configuration.
            commission_config: Commission scheme.
            margin_state: Margin tracking for the account.
        """
        self._order_manager = OrderManager(
            min_lot_size=min_lot_size,
            spread_config=spread_config or SpreadConfig(),
            commission_config=commission_config or CommissionConfig(),
            margin_state=margin_state,
        )
        self._spread_config = spread_config or SpreadConfig()
        self._commission_config = commission_config or CommissionConfig()
        self._margin_state = margin_state
        self._current_quotes: dict[str, Quote] = {}

    # ------------------------------------------------------------------
    #  Quote management
    # ------------------------------------------------------------------

    def set_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        timestamp: Optional[datetime] = None,
        volume: float = 0.0,
    ) -> Quote:
        """Set the current market quote for a symbol.

        Args:
            symbol: Trading pair.
            bid: Bid price.
            ask: Ask price.
            timestamp: Quote timestamp (defaults to now).
            volume: Reported volume.

        Returns:
            The created ``Quote``.
        """
        ts = timestamp or datetime.now(timezone.utc)
        quote = Quote(symbol=symbol, price=(bid + ask) / 2, bid=bid, ask=ask, timestamp=ts, volume=volume)
        self._current_quotes[symbol] = quote
        return quote

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Retrieve the latest quote for a symbol."""
        return self._current_quotes.get(symbol)

    @property
    def spread_config(self) -> SpreadConfig:
        """Current spread configuration."""
        return self._spread_config

    @spread_config.setter
    def spread_config(self, value: SpreadConfig) -> None:
        self._spread_config = value

    @property
    def margin_state(self) -> Optional[MarginState]:
        """Current margin state."""
        return self._margin_state

    @margin_state.setter
    def margin_state(self, value: MarginState) -> None:
        self._margin_state = value
        self._order_manager.margin_state = value

    # ------------------------------------------------------------------
    #  Order management (delegated to OrderManager)
    # ------------------------------------------------------------------

    @property
    def order_manager(self) -> OrderManager:
        """Underlying order manager for direct access."""
        return self._order_manager

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
        """Submit an order (delegated to the order manager).

        Args:
            symbol: Trading pair.
            order_type: Type of order.
            side: Buy or sell.
            quantity: Number of lots.
            price: Limit / stop price.
            stop_loss: Attached stop-loss.
            take_profit: Attached take-profit.

        Returns:
            The created ``Order``.
        """
        return self._order_manager.submit_order(
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def cancel_order(self, order_id: str) -> Order:
        """Cancel an order (delegated to the order manager)."""
        return self._order_manager.cancel_order(order_id)

    def amend_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Order:
        """Amend an order (delegated to the order manager)."""
        return self._order_manager.amend_order(
            order_id, price=price, quantity=quantity,
            stop_loss=stop_loss, take_profit=take_profit,
        )

    # ------------------------------------------------------------------
    #  Fill simulation
    # ------------------------------------------------------------------

    def simulate_market_fill(self, order_id: str) -> tuple[Order, FillRecord, TradeRecord]:
        """Fill a pending/open order at the current market price.

        Applies spread slippage based on the active quote and the configured
        spread settings.  For MARKET orders the mid-market price is used with
        the appropriate side spread applied.

        Args:
            order_id: Identifier of the order to fill.

        Returns:
            Tuple of ``(order, fill_record, trade_record)``.

        Raises:
            RuntimeError: If no quote is available for the order's symbol.
        """
        order = self._order_manager.get_order(order_id)
        if order is None:
            raise RuntimeError(f"Order {order_id} not found")

        quote = self._current_quotes.get(order.symbol)
        if quote is None:
            raise RuntimeError(
                f"No quote available for symbol {order.symbol}"
            )

        # Determine execution price with spread slippage
        exec_price = self._apply_spread_slippage(quote, order.side)

        return self._order_manager.fill_order(
            order_id=order_id,
            fill_price=exec_price,
            fill_type="manual",
        )

    def simulate_stop_trigger(
        self,
        order_id: str,
        triggered_price: float,
    ) -> tuple[Order, FillRecord, TradeRecord]:
        """Fill a STOP_LOSS or TAKE_PROFIT order that has been triggered.

        Args:
            order_id: Identifier of the triggered order.
            triggered_price: Price at which the stop was triggered.

        Returns:
            Tuple of ``(order, fill, trade)``.
        """
        order = self._order_manager.get_order(order_id)
        if order is None:
            raise RuntimeError(f"Order {order_id} not found")

        # Use a synthetic quote for the trigger price as the mid
        spread = self._spread_config.default_spread_pips
        if order.side == OrderSide.BUY:
            bid = triggered_price - spread / 2
            ask = triggered_price + spread / 2
        else:
            ask = triggered_price + spread / 2
            bid = triggered_price - spread / 2

        quote = Quote(
            symbol=order.symbol,
            price=triggered_price,
            bid=bid,
            ask=ask,
            timestamp=datetime.now(timezone.utc),
        )

        exec_price = self._apply_spread_slippage(quote, order.side)

        return self._order_manager.fill_order(
            order_id=order_id,
            fill_price=exec_price,
            fill_type="stop_trigger" if order.order_type == OrderType.STOP_LOSS else "tp_trigger",
        )

    def advance_quote(self, symbol: str) -> None:
        """Advance the current quote by random jitter.

        Simulates a small price movement using the configured spread jitter.
        This method is a convenience for testing – in production you would
        receive real-time quotes from a feed.

        Args:
            symbol: Trading pair to advance.

        Raises:
            RuntimeError: If no existing quote is found for the symbol.
        """
        quote = self._current_quotes.get(symbol)
        if quote is None:
            raise RuntimeError(f"No quote for symbol {symbol}")

        jitter = random.uniform(
            -self._spread_config.random_jitter_pips,
            self._spread_config.random_jitter_pips,
        )
        new_bid = quote.bid + jitter
        new_ask = quote.ask + jitter
        self.set_quote(symbol, new_bid, new_ask)

    # ------------------------------------------------------------------
    #  Trigger checking
    # ------------------------------------------------------------------

    def check_stop_loss_triggers(self) -> list[tuple[Order, FillRecord, TradeRecord]]:
        """Evaluate all open stop-loss orders against current quotes.

        Returns:
            List of ``(order, fill, trade)`` tuples for orders that
            were triggered and filled.
        """
        filled: list[tuple[Order, FillRecord, TradeRecord]] = []
        active_orders = self._order_manager.get_active_orders()

        for order in active_orders:
            if order.order_type != OrderType.STOP_LOSS:
                continue
            quote = self._current_quotes.get(order.symbol)
            if quote is None:
                continue

            # Stop-loss is triggered when price crosses the stop level
            if order.side == OrderSide.BUY:
                if quote.bid <= order.stop_loss:
                    result = self.simulate_stop_trigger(order.order_id, order.stop_loss)
                    filled.append(result)
            else:
                if quote.ask >= order.stop_loss:
                    result = self.simulate_stop_trigger(order.order_id, order.stop_loss)
                    filled.append(result)

        return filled

    def check_take_profit_triggers(self) -> list[tuple[Order, FillRecord, TradeRecord]]:
        """Evaluate all open take-profit orders against current quotes.

        Returns:
            List of ``(order, fill, trade)`` tuples for orders that
            were triggered and filled.
        """
        filled: list[tuple[Order, FillRecord, TradeRecord]] = []
        active_orders = self._order_manager.get_active_orders()

        for order in active_orders:
            if order.order_type != OrderType.TAKE_PROFIT:
                continue
            quote = self._current_quotes.get(order.symbol)
            if quote is None:
                continue

            if order.side == OrderSide.BUY:
                if quote.ask >= order.take_profit:
                    result = self.simulate_stop_trigger(order.order_id, order.take_profit)
                    filled.append(result)
            else:
                if quote.bid <= order.take_profit:
                    result = self.simulate_stop_trigger(order.order_id, order.take_profit)
                    filled.append(result)

        return filled

    def check_all_triggers(self) -> list[tuple[Order, FillRecord, TradeRecord]]:
        """Run all trigger checks and return combined results.

        Returns:
            List of all ``(order, fill, trade)`` tuples.
        """
        results: list[tuple[Order, FillRecord, TradeRecord]] = []
        results.extend(self.check_stop_loss_triggers())
        results.extend(self.check_take_profit_triggers())
        return results

    # ------------------------------------------------------------------
    #  Margin helpers
    # ------------------------------------------------------------------

    def reserve_margin(self, notional: float) -> float:
        """Reserve margin for a new position.

        Args:
            notional: Position notional value.

        Returns:
            Amount reserved.

        Raises:
            RuntimeError: If no margin state is configured.
        """
        if self._margin_state is None:
            raise RuntimeError("No margin state configured")
        return self._margin_state.reserve(notional)

    def release_margin(self, notional: float) -> float:
        """Release margin for a closed position.

        Args:
            notional: Position notional value.

        Returns:
            Amount released.
        """
        if self._margin_state is None:
            return 0.0
        return self._margin_state.release(notional)

    def update_equity(self, pnl: float) -> None:
        """Update total equity with realized P&L.

        Args:
            pnl: Realized profit or loss.
        """
        if self._margin_state is None:
            return
        self._margin_state.update_equity(pnl)

    # ------------------------------------------------------------------
    #  Commission helpers
    # ------------------------------------------------------------------

    def calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a fill.

        Args:
            quantity: Lots filled.
            price: Fill price.

        Returns:
            Commission amount.
        """
        return self._commission_config.calculate(quantity, price)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _apply_spread_slippage(self, quote: Quote, side: OrderSide) -> float:
        """Apply spread slippage to a quote for a given side.

        Args:
            quote: Current market quote.
            side: Order direction.

        Returns:
            Slipped execution price.
        """
        return self._spread_config.apply_spread(quote.price, side)
