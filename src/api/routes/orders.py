from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

router = APIRouter()


class OrderRequest(BaseModel):
    symbol: str
    side: str
    order_type: str = "MARKET"
    quantity: float = Field(..., gt=0)
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None


class OrderResponse(BaseModel):
    order_id: str
    symbol: str
    side: str
    type: str
    quantity: float
    price: float | None
    status: str
    created_at: str


class OrderListResponse(BaseModel):
    orders: list[OrderResponse]


@router.post("/orders", response_model=OrderResponse)
def create_order(req: OrderRequest):
    return OrderResponse(
        order_id="uuid-here", symbol=req.symbol, side=req.side, type=req.order_type,
        quantity=req.quantity, price=req.price, status="PENDING", created_at="2024-01-01T00:00:00Z",
    )


@router.get("/orders", response_model=OrderListResponse)
def list_orders(
    status: str | None = Query(None),
    symbol: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    return OrderListResponse(orders=[])


@router.get("/orders/{order_id}", response_model=OrderResponse)
def get_order(order_id: str):
    return OrderResponse(
        order_id=order_id, symbol="", side="", type="", quantity=0, price=None, status="PENDING",
        created_at="2024-01-01T00:00:00Z",
    )


@router.post("/orders/{order_id}/cancel")
def cancel_order(order_id: str):
    return {"status": "cancelled", "order_id": order_id}
