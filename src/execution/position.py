from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class Position(BaseModel):
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_drawdown: float = 0.0
    entry_time: datetime = datetime.now()
    exit_time: Optional[datetime] = None
    strategy_id: Optional[str] = None
    order_ids: list[str] = []
