from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter()


class StrategyInfo(BaseModel):
    name: str
    description: str
    params: dict
    enabled: bool = False


class StrategyListResponse(BaseModel):
    strategies: list[StrategyInfo]


@router.get("/strategies", response_model=StrategyListResponse)
def list_strategies():
    return StrategyListResponse(
        strategies=[
            StrategyInfo(name="ma_crossover", description="Moving Average Crossover", params={"fast": 10, "slow": 50}),
            StrategyInfo(name="mean_reversion", description="Bollinger Band Mean Reversion", params={"period": 20, "std": 2}),
            StrategyInfo(name="rsi", description="RSI Divergence", params={"period": 14, "overbought": 70, "oversold": 30}),
            StrategyInfo(name="macd", description="MACD Signal", params={"fast": 12, "slow": 26, "signal": 9}),
            StrategyInfo(name="bollinger_bands", description="Bollinger Bands Breakout", params={"period": 20, "std": 2}),
        ]
    )


@router.post("/strategies/{name}/enable")
def enable_strategy(name: str):
    return {"status": "enabled", "strategy": name}


@router.post("/strategies/{name}/disable")
def disable_strategy(name: str):
    return {"status": "disabled", "strategy": name}
