from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

router = APIRouter()


class BacktestRequest(BaseModel):
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = Field(100000, gt=0)
    params: dict[str, float] = {}


class BacktestMetrics(BaseModel):
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    calmar_ratio: float


class BacktestResponse(BaseModel):
    metrics: BacktestMetrics
    equity_curve: list[float]
    trades: list[dict]


@router.post("/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest):
    return BacktestResponse(
        metrics=BacktestMetrics(
            total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, max_drawdown=0.0, win_rate=0.0,
            profit_factor=0.0, total_trades=0, calmar_ratio=0.0,
        ),
        equity_curve=[], trades=[],
    )


@router.get("/backtest/results")
def get_backtest_results(limit: int = Query(10, ge=1, le=100)):
    return {"results": []}
