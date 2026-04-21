"""Pydantic models and exceptions for the risk management module."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_serializer


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskMode(str, Enum):
    """Risk tolerance mode that drives position sizing and limits."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ValidationResult(str, Enum):
    """Result of a pre-trade risk check."""

    PASS = "pass"
    FAIL = "fail"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RiskLimitExceeded(Exception):
    """Raised when a pre-trade risk check fails."""

    def __init__(self, reason: str, value: float, limit: float) -> None:
        self.reason = reason
        self.value = value
        self.limit = limit
        super().__init__(
            f"{reason}: {value:.4f} exceeds limit {limit:.4f}"
        )


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

class RiskParams(BaseModel):
    """Configurable risk parameters for a trading account.

    Attributes:
        risk_mode: Overall risk tolerance mode.
        max_position_size_pct: Maximum single-position size as % of equity.
        max_total_exposure_pct: Maximum total portfolio exposure as % of equity.
        max_correlation: Maximum allowed correlation between positions (0-1).
        max_drawdown_pct: Maximum drawdown from peak equity as %.
        max_daily_loss_pct: Maximum daily loss as % of starting equity.
        margin_requirement_pct: Margin required per position as %.
        kelly_fraction: Fraction of full Kelly bet (e.g. 0.25 for quarter-Kelly).
        atr_multiplier: Multiplier used in ATR-based stop loss / sizing.
        risk_per_trade_pct: Default risk per trade as % of equity (0-1).
    """

    risk_mode: RiskMode = RiskMode.MODERATE
    max_position_size_pct: float = Field(default=0.05, ge=0, le=1)
    max_total_exposure_pct: float = Field(default=0.20, ge=0, le=1)
    max_correlation: float = Field(default=0.8, ge=0, le=1)
    max_drawdown_pct: float = Field(default=0.10, ge=0, le=1)
    max_daily_loss_pct: float = Field(default=0.02, ge=0, le=1)
    margin_requirement_pct: float = Field(default=0.02, ge=0, le=1)
    kelly_fraction: float = Field(default=0.25, ge=0, le=1)
    atr_multiplier: float = Field(default=1.5, gt=0)
    risk_per_trade_pct: float = Field(default=0.02, ge=0, le=1)

    @property
    def position_size(self) -> float:
        """Return effective max position size based on risk mode."""
        multipliers: dict[RiskMode, float] = {
            RiskMode.CONSERVATIVE: 0.5,
            RiskMode.MODERATE: 0.75,
            RiskMode.AGGRESSIVE: 1.0,
        }
        return self.max_position_size_pct * multipliers[self.risk_mode]


class PositionInfo(BaseModel):
    """Snapshot of an open position.

    Attributes:
        symbol: Trading pair (e.g. "EURUSD").
        side: "long" or "short".
        entry_price: Price at entry.
        quantity: Number of units / lots.
        stop_loss: Current stop-loss price (None if unset).
        take_profit: Current take-profit price (None if unset).
        unrealized_pnl: Current unrealized P&L.
        peak_equity: Running peak equity for drawdown tracking.
        entry_time: Timestamp of position entry.
    """

    symbol: str
    side: str
    entry_price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    peak_equity: float = 0.0
    entry_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_serializer("entry_time")
    def _serialize_entry_time(self, value: datetime) -> str:
        return value.isoformat()


class RiskAlert(BaseModel):
    """Record of a risk event / alert.

    Attributes:
        timestamp: When the alert was generated.
        severity: "info", "warning", "critical".
        category: What triggered the alert.
        message: Human-readable description.
        current_value: The measured value at alert time.
        threshold: The threshold that was breached (or target for info).
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: str = "info"
    category: str
    message: str
    current_value: float
    threshold: float

    @field_serializer("timestamp")
    def _serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()


class PreTradeResult(BaseModel):
    """Result of a full pre-trade risk validation.

    Attributes:
        passed: Overall pass / fail.
        checks: Individual check results.
    """

    passed: bool
    checks: list[str] = Field(default_factory=list)
