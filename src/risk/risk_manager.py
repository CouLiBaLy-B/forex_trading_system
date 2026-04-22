"""Core risk management for forex trading."""

import threading
from datetime import datetime, timezone

import numpy as np

from .models import (
    PositionInfo,
    RiskAlert,
    RiskLimitExceeded,
    RiskMode,
    RiskParams,
    ValidationResult,
)
from .position_sizer import PositionSizer
from ..indicators.base import PriceSeries


class RiskManager:
    """Central risk management engine."""

    def __init__(self, params: RiskParams, account_id: str = "default"):
        self.params = params
        self.account_id = account_id
        self.sizer = PositionSizer(params)
        self._positions: dict[str, PositionInfo] = {}
        self._daily_pnl: float = 0.0
        self._alert_history: list[RiskAlert] = []
        self._peak_equity: dict[str, float] = {}
        self._correlation_matrix: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    # ---- Position management ----

    def open_position(self, symbol: str, side: str, price: float, quantity: float,
                     stop_loss: float | None = None, take_profit: float | None = None,
                     current_equity: float | None = None) -> PositionInfo:
        with self._lock:
            position = PositionInfo(
                symbol=symbol, side=side, entry_price=price,
                quantity=quantity, stop_loss=stop_loss, take_profit=take_profit,
            )
            self._positions[symbol] = position
            equity = self._get_equity(current_equity, price, quantity)
            self._update_peak_equity(equity)
            return position

    def _get_equity(self, current_equity: float | None, price: float, quantity: float) -> float:
        if current_equity is not None:
            return current_equity
        notional = abs(price * quantity)
        return notional  # fallback to notional if equity unknown

    def _update_peak_equity(self, current_equity: float) -> None:
        if current_equity > self._peak_equity.get(self.account_id, 0):
            self._peak_equity[self.account_id] = current_equity

    def close_position(self, symbol: str, exit_price: float, current_equity: float | None = None) -> float:
        with self._lock:
            position = self._positions.pop(symbol)
            if position.side == "long":
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - exit_price) * position.quantity
            self._daily_pnl += pnl
            equity = self._get_equity(current_equity, exit_price, 0)
            self._update_peak_equity(equity)
            return pnl

    def get_positions(self) -> dict[str, PositionInfo]:
        with self._lock:
            return dict(self._positions)

    def _get_position_notional(self, position: PositionInfo) -> float:
        """Approximate notional value."""
        return abs(position.entry_price * position.quantity)

    # ---- Pre-trade checks ----

    def check_position_size(self, symbol: str, price: float, quantity: float,
                            current_equity: float) -> bool:
        """Check if position value / equity is within limits."""
        if current_equity <= 0:
            return False
        max_size = self.params.position_size * current_equity
        position_value = abs(price * quantity)
        return position_value <= max_size

    def check_exposure(self, symbol: str, price: float, quantity: float,
                       current_equity: float) -> bool:
        """Check total portfolio exposure (notional) vs equity."""
        with self._lock:
            notional = current_equity + abs(price * quantity)
            for pos in self._positions.values():
                notional += self._get_position_notional(pos)
            return notional / current_equity <= (1 + self.params.max_total_exposure_pct)

    def check_correlation(self, symbol: str) -> float:
        """Get max pairwise correlation of new position with existing positions."""
        if not self._correlation_matrix:
            return 0.0
        max_corr = 0.0
        for sym, matrix in self._correlation_matrix.items():
            if sym in self._positions:
                max_corr = max(max_corr, abs(matrix[self._positions[sym].symbol]))
        return max_corr

    def check_drawdown(self, current_equity: float) -> bool:
        """Check drawdown from peak equity."""
        peak = self._peak_equity.get(self.account_id, 0)
        if peak == 0:
            return True
        drawdown = (peak - current_equity) / peak
        return drawdown <= self.params.max_drawdown_pct

    def check_daily_loss(self, current_equity: float) -> bool:
        """Check daily P&L loss vs daily loss limit."""
        if current_equity <= 0:
            return False
        return abs(self._daily_pnl) / current_equity <= self.params.max_daily_loss_pct

    def check_margin(self, symbol: str, price: float, quantity: float,
                     current_equity: float) -> bool:
        """Check margin requirements."""
        margin_for_new = abs(price * quantity) * self.params.margin_requirement_pct
        required = margin_for_new
        for sym, pos in self._positions.items():
            required += self._get_position_notional(pos) * self.params.margin_requirement_pct
        return current_equity >= required

    def pre_trade_check(self, symbol: str, side: str, price: float, quantity: float,
                        current_equity: float) -> ValidationResult:
        """Run all pre-trade checks."""
        try:
            if not self.check_position_size(symbol, price, quantity, current_equity):
                raise RiskLimitExceeded("Position size", abs(price * quantity),
                                        self.params.position_size * current_equity)
            if not self.check_exposure(symbol, price, quantity, current_equity):
                raise RiskLimitExceeded("Total exposure", abs(price * quantity) / current_equity,
                                        self.params.max_total_exposure_pct)
            if not self.check_drawdown(current_equity):
                raise RiskLimitExceeded("Drawdown", self._peak_equity.get(self.account_id, 0) / current_equity,
                                        1 - self.params.max_drawdown_pct)
            if not self.check_daily_loss(current_equity):
                raise RiskLimitExceeded("Daily loss", abs(self._daily_pnl),
                                        self.params.max_daily_loss_pct * current_equity)
            if not self.check_margin(symbol, price, quantity, current_equity):
                raise RiskLimitExceeded("Margin", self._get_position_notional(PositionInfo(symbol=symbol, side=side, entry_price=price, quantity=quantity)) * self.params.margin_requirement_pct,
                                        current_equity)
        except RiskLimitExceeded as e:
            self._record_alert(severity="critical", category="pre_trade",
                             message=str(e), current_value=e.value, limit=e.limit)
            return ValidationResult.FAIL
        return ValidationResult.PASS

    # ---- Dynamic SL / TP ----

    def compute_stop_loss_take_profit(self, prices: PriceSeries, side: str, entry_price: float,
                                      atr_value: float) -> tuple[float | None, float | None]:
        """Compute stop / take-profit from ATR."""
        multiplier = self.params.atr_multiplier
        if side == "long":
            return entry_price - multiplier * atr_value, entry_price + multiplier * atr_value * 1.5
        return entry_price + multiplier * atr_value, entry_price - multiplier * atr_value * 1.5

    # ---- Alerts ----

    def _record_alert(self, severity: str, category: str, message: str,
                      current_value: float, limit: float) -> RiskAlert:
        alert = RiskAlert(severity=severity, category=category, message=message,
                        current_value=current_value, threshold=limit)
        self._alert_history.append(alert)
        return alert

    def get_alerts(self, severity: str | None = None, category: str | None = None) -> list[RiskAlert]:
        alerts = self._alert_history
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]
        return alerts

    def filter_alerts(self, min_severity: str, category: str | None = None) -> list[RiskAlert]:
        severity_order = {"info": 0, "warning": 1, "critical": 2}
        min_level = severity_order.get(min_severity, 0)
        alerts = [a for a in self._alert_history if severity_order.get(a.severity, 0) >= min_level]
        if category:
            alerts = [a for a in alerts if a.category == category]
        return alerts
