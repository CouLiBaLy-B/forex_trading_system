from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class SystemMetrics(BaseModel):
    active_strategies: int
    open_positions: int
    daily_pnl: float
    total_equity: float
    system_status: str
    uptime_seconds: float


@router.get("/metrics", response_model=SystemMetrics)
def get_system_metrics():
    return SystemMetrics(
        active_strategies=0, open_positions=0, daily_pnl=0.0,
        total_equity=0.0, system_status="running", uptime_seconds=0.0,
    )
