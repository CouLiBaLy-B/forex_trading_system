"""Execution engine for forex trading.

Exports:
    * ``OrderManager`` – order lifecycle, validation, order book management.
    * ``PaperTradingEngine`` – paper/simulated trading with spread slippage.
    * ``FillRecord`` – pydantic model for fill events.
    * ``TradeRecord`` – pydantic model for post-fill trade summaries.
    * ``Order`` – pydantic model for orders.
    * ``OrderBook`` – pydantic model for active order collections.
    * ``OrderType`` – enum of supported order types (MARKET, LIMIT, STOP_LOSS, TAKE_PROFIT).
    * ``OrderSide`` – enum (BUY, SELL).
    * ``OrderStatus`` – enum for order lifecycle states.
    * ``CommissionConfig``, ``SpreadConfig``, ``MarginState`` – pydantic helpers.
    * ``OrderValidationError``, ``MarginExceededError``, ``OrderNotFound``.
"""

from .models import (
    CommissionConfig,
    FillRecord,
    MarginState,
    MarginExceededError,
    Order,
    OrderBook,
    OrderNotFound,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
    Quote,
    SpreadConfig,
    TradeRecord,
)
from .order_manager import OrderManager
from .paper_engine import PaperTradingEngine

__all__ = [
    "CommissionConfig",
    "FillRecord",
    "MarginExceededError",
    "MarginState",
    "Order",
    "OrderBook",
    "OrderManager",
    "OrderNotFound",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "OrderValidationError",
    "PaperTradingEngine",
    "Quote",
    "SpreadConfig",
    "TradeRecord",
]
