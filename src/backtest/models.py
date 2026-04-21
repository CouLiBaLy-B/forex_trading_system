"""Pydantic models for backtesting configuration, results, and trade records."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PositionSide(str, Enum):
    """Direction of a position."""
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(str, Enum):
    """Reason a trade was closed."""
    SIGNAL = "SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TIME_STOP = "TIME_STOP"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class BacktestConfig(BaseModel):
    """Configuration for a backtest run.

    All monetary values are in account base currency (typically USD).
    Percentages are expressed as decimals (e.g. 0.02 for 2 %).
    """

    # ---- core parameters ----
    initial_capital: float = Field(default=100_000.0, gt=0)
    instrument: str = Field(default="EUR/USD", min_length=1)
    strategy_name: str = Field(default="", min_length=0)

    # ---- market microstructure ----
    spread: float = Field(default=0.0001, ge=0)       # fixed spread in price units
    slippage: float = Field(default=0.0, ge=0)         # additional slippage in price units
    commission_rate: float = Field(default=0.0, ge=0, le=1)  # fraction of notional
    commission_mode: str = Field(default="notional")   # "notional" | "per_unit"
    commission_per_unit: float = Field(default=0.0, ge=0)

    # ---- sizing ----
    risk_per_trade: float = Field(default=0.02, gt=0, le=1)  # fraction of equity
    max_position_size: float = Field(default=1.0, gt=0, le=1)  # fraction of equity
    leverage: float = Field(default=1.0, gt=0)

    # ---- validation parameters ----
    warm_up_bars: int = Field(default=0, ge=0)
    lookback: int = Field(default=0, ge=0)

    # ---- walking-forward ----
    walk_forward: WalkForwardConfig | None = Field(default=None)

    # ---- misc ----
    allow_partial_fills: bool = Field(default=False)

    model_config = {"frozen": True}

    @computed_field  # type: ignore[misc]
    @property
    def effective_spread(self) -> float:
        """Spread + slippage in price units."""
        return self.spread + self.slippage


class WalkForwardConfig(BaseModel):
    """Configuration for walking-forward (walk-forward) validation."""

    in_sample_period: int = Field(default=252, gt=0)   # bars in-sample
    out_of_sample_period: int = Field(default=63, gt=0)  # bars out-of-sample
    num_folds: int = Field(default=5, gt=0)
    step: int = Field(default=63, gt=0)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

class BacktestTrade(BaseModel):
    """Single simulated trade produced by the backtest engine."""

    index: int = -1
    instrument: str = ""
    side: PositionSide = PositionSide.LONG
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    exit_reason: ExitReason = ExitReason.SIGNAL
    max_drawdown_during: float = 0.0
    peak_equity_during: float = 0.0
    holding_bars: int = 0
    strategy_name: str = ""
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[misc]
    @property
    def holding_period_seconds(self) -> float:
        """Duration of the trade in seconds."""
        if self.entry_time and self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return 0.0

    @computed_field  # type: ignore[misc]
    @property
    def notional(self) -> float:
        """Notional value of the trade (entry side)."""
        return abs(self.entry_price * self.quantity)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class BacktestResult(BaseModel):
    """Complete backtesting output with all performance metrics.

    All ratio metrics (Sharpe, Sortino, Calmar) annualise returns assuming
    252 trading days per year.  For tick data the user should set
    ``trading_days_per_year`` accordingly.
    """

    # ---- identity ----
    strategy_name: str = ""
    instrument: str = ""
    start_date: str = ""
    end_date: str = ""
    model_config = {"frozen": True}

    # ---- trade counts ----
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    total_winning_pnl: float = 0.0
    total_losing_pnl: float = 0.0

    # ---- return metrics ----
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    net_return: float = 0.0  # after commissions
    annualized_return: float = 0.0
    total_days: float = 0.0

    # ---- risk metrics ----
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_start: str = ""
    max_drawdown_end: str = ""
    largest_drawdown_day: float = 0.0
    avg_drawdown: float = 0.0
    avg_drawdown_pct: float = 0.0

    # ---- ratio metrics ----
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    omega_ratio: float = 0.0

    # ---- trade-level statistics ----
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_period: float = 0.0
    avg_holding_bars: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    worst_consecutive_losses: int = 0
    best_consecutive_wins: int = 0

    # ---- commission & slippage ----
    total_commission: float = 0.0
    total_slippage_cost: float = 0.0

    # ---- benchmark ----
    benchmark_return: float = 0.0
    benchmark_sharpe: float = 0.0
    excess_return: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0

    # ---- internal / raw data ----
    equity_curve: list[float] = Field(default_factory=list)
    benchmark_curve: list[float] | None = None
    drawdown_curve: list[float] = Field(default_factory=list)
    trade_log: list[BacktestTrade] = Field(default_factory=list)
    daily_returns: list[float] = Field(default_factory=list)

    # ---- metadata ----
    config: dict[str, Any] = Field(default_factory=dict)
    trading_days_per_year: int = 252
    strategy_metadata: dict[str, Any] = Field(default_factory=dict)

    # ---- computed fields ----

    @computed_field  # type: ignore[misc]
    @property
    def calmar_computed(self) -> float:
        """Calmar ratio: annualised return / max drawdown."""
        if self.max_drawdown_pct == 0:
            return 0.0
        return self.annualized_return / abs(self.max_drawdown_pct)

    @computed_field  # type: ignore[misc]
    @property
    def profit_loss_ratio(self) -> float:
        """Average win / average loss."""
        if self.avg_loss == 0:
            return 0.0
        return self.avg_win / abs(self.avg_loss)

    @computed_field  # type: ignore[misc]
    @property
    def total_pnl(self) -> float:
        """Total gross P&L (before commission)."""
        return self.total_winning_pnl + self.total_losing_pnl

    @computed_field  # type: ignore[misc]
    @property
    def sortino_computed(self) -> float:
        """Sortino ratio: annualised return / downside deviation."""
        downside = self._downside_deviation
        if downside == 0:
            return 0.0
        return self.annualized_return / downside

    def _downside_deviation(self) -> float:
        """Downside deviation of daily returns (population)."""
        if not self.daily_returns:
            return 0.0
        downside = [(r - 0) ** 2 for r in self.daily_returns if r < 0]
        if not downside:
            return 0.0
        return (sum(downside) / len(self.daily_returns)) ** 0.5


class BacktestFoldResult(BaseModel):
    """Results from a single walking-forward fold."""

    fold_index: int
    in_sample_period: tuple[int, int]
    out_of_sample_period: tuple[int, int]
    result: BacktestResult
    metrics: dict[str, float] = Field(default_factory=dict)
