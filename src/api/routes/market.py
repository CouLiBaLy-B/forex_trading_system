from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

router = APIRouter()


class QuoteResponse(BaseModel):
    symbol: str
    bid: float
    ask: float
    last: float
    spread: float
    timestamp: str


class OHLCVResponse(BaseModel):
    symbol: str
    interval: str
    candles: list[dict]


@router.get("/quotes", response_model=QuoteResponse)
def get_quote(symbol: str = Query(..., description="Trading pair, e.g. EUR/USD")):
    return QuoteResponse(
        symbol=symbol, bid=1.0850, ask=1.0852, last=1.0851, spread=0.0002, timestamp="2024-01-01T00:00:00Z",
    )


@router.get("/ohlcv", response_model=OHLCVResponse)
def get_ohlcv(
    symbol: str = Query(...),
    interval: str = Query("1h", description="1m, 5m, 15m, 1h, 4h, 1d"),
    limit: int = Query(100, ge=1, le=1000),
):
    return OHLCVResponse(symbol=symbol, interval=interval, candles=[])
