"""Risk management module for forex trading.

Exports:
    * ``RiskManager`` – pre-trade validation, dynamic SL/TP, risk alerts.
    * ``PositionSizer`` – fixed-fraction, Kelly, ATR sizing strategies.
    * ``RiskParams`` – pydantic configuration model.
    * ``RiskMode`` – conservative / moderate / aggressive risk mode enum.
    * ``RiskAlert``, ``PositionInfo``, ``PreTradeResult``, ``RiskLimitExceeded``.
"""

from .models import (
    PositionInfo,
    PreTradeResult,
    RiskAlert,
    RiskLimitExceeded,
    RiskMode,
    RiskParams,
    ValidationResult,
)
from .position_sizer import PositionSizer
from .risk_manager import RiskManager

__all__ = [
    "PositionInfo",
    "PositionSizer",
    "PreTradeResult",
    "RiskAlert",
    "RiskLimitExceeded",
    "RiskManager",
    "RiskMode",
    "RiskParams",
    "ValidationResult",
]
