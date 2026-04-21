from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class PositionInfo(BaseModel):
    symbol: str
    side: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    pnl_pct: float


class PortfolioSnapshot(BaseModel):
    total_equity: float
    cash: float
    margin_used: float
    positions: list[PositionInfo]
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float


@router.get("/portfolio", response_model=PortfolioSnapshot)
def get_portfolio():
    return PortfolioSnapshot(
        total_equity=100000, cash=50000, margin_used=0, positions=[],
        total_pnl=0, daily_pnl=0, unrealized_pnl=0,
    )


class TradeRecord(BaseModel):
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    entry_time: str
    exit_time: str | None


class TradeListResponse(BaseModel):
    trades: list[TradeRecord]


@router.get("/portfolio/trades", response_model=TradeListResponse)
def get_trades(
    symbol: str | None = None,
    limit: int = 50,
):
    return TradeListResponse(trades=[])
