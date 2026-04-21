import uuid
from .order import Order, OrderType, OrderSide, OrderStatus
from .position import Position
from market_data.service import MarketDataService

class ExecutionEngine:
    def __init__(self):
        self.orders: list[Order] = []
        self.positions: dict[str, Position] = {}
        self.market_service = MarketDataService()

    def create_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                     quantity: float, price: float = None, strategy_id: str = None) -> Order:
        order = Order(
            symbol=symbol, side=side, order_type=order_type,
            quantity=quantity, price=price, status=OrderStatus.PENDING,
            order_id=str(uuid.uuid4())[:8], strategy_id=strategy_id
        )
        self.orders.append(order)
        return order

    def submit_order(self, order: Order) -> Order:
        if order.status != OrderStatus.PENDING:
            raise ValueError(f"Order {order.order_id} is not pending")
        order.status = OrderStatus.SUBMITTED
        return self._execute(order)

    def _execute(self, order: Order) -> Order:
        if order.order_type == OrderType.MARKET:
            quote = self.market_service.get_quote(order.symbol)
            fill_price = quote.bid if order.side == OrderSide.SELL else quote.ask
            order.status = OrderStatus.FILLED
            order.filled_price = fill_price
            order.filled_quantity = order.quantity
        elif order.order_type == OrderType.LIMIT:
            if (order.side == OrderSide.BUY and order.price >= order.filled_price) or \
               (order.side == OrderSide.SELL and order.price <= order.filled_price):
                order.status = OrderStatus.FILLED
                order.filled_price = order.price
                order.filled_quantity = order.quantity
        return order

    def open_position(self, order: Order) -> Position:
        position = Position(
            symbol=order.symbol, side=order.side.value,
            quantity=order.quantity, entry_price=order.filled_price or order.price,
            strategy_id=order.strategy_id
        )
        self.positions[order.symbol] = position
        return position

    def close_position(self, symbol: str) -> Position | None:
        position = self.positions.pop(symbol, None)
        if position:
            position.exit_time = datetime.now()
        return position

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def get_all_positions(self) -> list[Position]:
        return list(self.positions.values())

    def cancel_order(self, order_id: str) -> Order | None:
        for order in self.orders:
            if order.order_id == order_id and order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                order.status = OrderStatus.CANCELLED
                return order
        return None

    def get_order_history(self, strategy_id: str = None) -> list[Order]:
        orders = self.orders
        if strategy_id:
            orders = [o for o in orders if o.strategy_id == strategy_id]
        return sorted(orders, key=lambda o: o.created_at, reverse=True)
