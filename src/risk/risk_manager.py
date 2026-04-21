"""RiskManager – pre-trade validation, dynamic SL/TP, and risk alerts."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

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


class RiskManager:
    """Central risk engine for forex trading.

    Responsibilities:
        * **Pre-trade checks** – position size, exposure, correlation,
          drawdown, daily loss, margin (all return ``Pass`` or ``Fail``).
        * **Dynamic SL/TP** – ATR-based stop-loss and take-profit.
        * **Risk alerts** – event-driven alert recording with severity levels.

    All numeric limits are expressed as *fractions* of equity (0-1).
    """

    def __init__(self, risk_params: Optional[RiskParams] = None) -> None:
        """Initialise with optional risk configuration.

        Args:
            risk_params: Risk parameters. Falls back to defaults for
                moderate mode when omitted.
        """
        self.params = risk_params or RiskParams(risk_mode=RiskMode.MODERATE)
        self._peak_equity: float = 0.0
        self._daily_start_equity: float = 0.0
        self._daily_pnl: float = 0.0
        self._alert_history: list[RiskAlert] = []
        self._positions: list[PositionInfo] = []
        self._correlation_matrix: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    #  Pre-trade checks  – each returns Pass/Fail with a reason on Fail
    # ------------------------------------------------------------------

    def check_position_size(
        self,
        position_value: float,
        account_equity: float,
    ) -> PreTradeResult:
        """Validate that a single position does not exceed the allowed size.

        Args:
            position_value: Notional value of the proposed position.
            account_equity: Current account equity.

        Returns:
            PreTradeResult with passed flag and check name on failure.
        """
        effective_limit = self.params.position_size
        ratio = position_value / account_equity if account_equity > 0 else 0.0

        if ratio > effective_limit:
            reason = (
                f"position_size: {ratio:.4f} exceeds limit {effective_limit:.4f}"
            )
            return PreTradeResult(passed=False, checks=[reason])
        return PreTradeResult(passed=True)

    def check_exposure(
        self,
        proposed_position_value: float,
        account_equity: float,
    ) -> PreTradeResult:
        """Validate total portfolio exposure stays within limits.

        Args:
            proposed_position_value: Notional of the *new* position.
            account_equity: Current account equity.

        Returns:
            PreTradeResult with passed flag and check name on failure.
        """
        current_exposure = PositionSizer.aggregate_exposure(
            self._positions, account_equity
        )
        new_exposure = (current_exposure * account_equity + proposed_position_value) / account_equity

        if new_exposure > self.params.max_total_exposure_pct:
            reason = (
                f"exposure: {new_exposure:.4f} exceeds limit "
                f"{self.params.max_total_exposure_pct:.4f}"
            )
            return PreTradeResult(passed=False, checks=[reason])
        return PreTradeResult(passed=True)

    def check_correlation(
        self,
        new_symbol: str,
        account_equity: float,
    ) -> PreTradeResult:
        """Check correlation between the proposed symbol and existing positions.

        Iterates open positions and looks up the stored correlation coefficient.
        If any correlation exceeds ``max_correlation``, the check fails.

        Args:
            new_symbol: Symbol of the proposed trade.
            account_equity: Current account equity.

        Returns:
            PreTradeResult with passed flag and check name on failure.
        """
        for pos in self._positions:
            corr = self._correlation_matrix.get(
                tuple(sorted((pos.symbol, new_symbol))), 0.0
            )
            if corr > self.params.max_correlation:
                reason = (
                    f"correlation({pos.symbol},{new_symbol}): "
                    f"{corr:.4f} exceeds max {self.params.max_correlation:.4f}"
                )
                return PreTradeResult(passed=False, checks=[reason])
        return PreTradeResult(passed=True)

    def check_drawdown(self, current_equity: float) -> PreTradeResult:
        """Validate current drawdown from peak equity.

        Args:
            current_equity: Current account equity.

        Returns:
            PreTradeResult with passed flag and check name on failure.
        """
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity <= 0:
            return PreTradeResult(passed=True)

        drawdown = (self._peak_equity - current_equity) / self._peak_equity

        if drawdown > self.params.max_drawdown_pct:
            reason = (
                f"drawdown: {drawdown:.4f} exceeds limit "
                f"{self.params.max_drawdown_pct:.4f}"
            )
            return PreTradeResult(passed=False, checks=[reason])
        return PreTradeResult(passed=True)

    def check_daily_loss(self, current_equity: float) -> PreTradeResult:
        """Validate today's P&L against the daily loss limit.

        Args:
            current_equity: Current account equity.

        Returns:
            PreTradeResult with passed flag and check name on failure.
        """
        if self._daily_start_equity <= 0:
            return PreTradeResult(passed=True)

        daily_loss = (self._daily_start_equity - current_equity) / self._daily_start_equity

        if daily_loss > self.params.max_daily_loss_pct:
            reason = (
                f"daily_loss: {daily_loss:.4f} exceeds limit "
                f"{self.params.max_daily_loss_pct:.4f}"
            )
            return PreTradeResult(passed=False, checks=[reason])
        return PreTradeResult(passed=True)

    def check_margin(
        self,
        position_value: float,
        account_equity: float,
    ) -> PreTradeResult:
        """Validate that margin requirement is satisfied.

        Args:
            position_value: Notional of the proposed position.
            account_equity: Current account equity.

        Returns:
            PreTradeResult with passed flag and check name on failure.
        """
        required_margin = position_value * self.params.margin_requirement_pct

        if required_margin > account_equity:
            reason = (
                f"margin: required {required_margin:.2f} > equity {account_equity:.2f}"
            )
            return PreTradeResult(passed=False, checks=[reason])
        return PreTradeResult(passed=True)

    # ------------------------------------------------------------------
    #  Full pre-trade validation  – runs every check
    # ------------------------------------------------------------------

    def pre_trade_check(
        self,
        position_value: float,
        account_equity: float,
        proposed_symbol: Optional[str] = None,
    ) -> PreTradeResult:
        """Run *all* pre-trade checks and return aggregated result.

        Args:
            position_value: Notional of the proposed position.
            account_equity: Current account equity.
            proposed_symbol: Symbol for correlation check (optional).

        Returns:
            PreTradeResult with passed flag and *all* failure reasons.
        """
        all_failures: list[str] = []

        results = [
            self.check_position_size(position_value, account_equity),
            self.check_exposure(position_value, account_equity),
            self.check_drawdown(account_equity),
            self.check_daily_loss(account_equity),
            self.check_margin(position_value, account_equity),
        ]

        if proposed_symbol:
            results.append(self.check_correlation(proposed_symbol, account_equity))

        for r in results:
            if not r.passed:
                all_failures.extend(r.checks)

        return PreTradeResult(passed=not all_failures, checks=all_failures)

    # ------------------------------------------------------------------
    #  Dynamic stop-loss / take-profit
    # ------------------------------------------------------------------

    def calc_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr: float,
    ) -> float:
        """Delegate to ``PositionSizer.calc_dynamic_stop_loss``.

        Args:
            entry_price: Entry price.
            side: ``"long"`` or ``"short"``.
            atr: Average True Range.

        Returns:
            Stop-loss price.
        """
        return PositionSizer.calc_dynamic_stop_loss(
            entry_price, side, atr, self.params.atr_multiplier
        )

    def calc_take_profit(
        self,
        entry_price: float,
        side: str,
        atr: float,
        rr_ratio: float = 2.0,
    ) -> float:
        """Delegate to ``PositionSizer.calc_dynamic_take_profit``.

        Args:
            entry_price: Entry price.
            side: ``"long"`` or ``"short"``.
            atr: Average True Range.
            rr_ratio: Risk-reward ratio (default 2.0).

        Returns:
            Take-profit price.
        """
        return PositionSizer.calc_dynamic_take_profit(
            entry_price, side, atr, self.params.atr_multiplier, rr_ratio
        )

    # ------------------------------------------------------------------
    #  Risk alerts system
    # ------------------------------------------------------------------

    def emit_alert(
        self,
        severity: str,
        category: str,
        message: str,
        current_value: float,
        threshold: float,
    ) -> RiskAlert:
        """Record and return a ``RiskAlert``.

        Args:
            severity: ``"info"``, ``"warning"``, or ``"critical"``.
            category: Short category label.
            message: Human-readable description.
            current_value: The measured value at alert time.
            threshold: The threshold that was breached / targeted.

        Returns:
            The created ``RiskAlert``.
        """
        alert = RiskAlert(
            severity=severity,
            category=category,
            message=message,
            current_value=current_value,
            threshold=threshold,
        )
        self._alert_history.append(alert)
        return alert

    def get_alerts(
        self,
        severity: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[RiskAlert]:
        """Retrieve alerts with optional filters.

        Args:
            severity: Filter by severity level.
            since: Filter alerts on or after this timestamp.

        Returns:
            List of matching ``RiskAlert`` objects.
        """
        result = self._alert_history
        if severity is not None:
            result = [a for a in result if a.severity == severity]
        if since is not None:
            result = [a for a in result if a.timestamp >= since]
        return result

    # ------------------------------------------------------------------
    #  Position management
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
    ) -> PositionInfo:
        """Record a new open position and update peak equity.

        Args:
            symbol: Trading pair.
            side: ``"long"`` or ``"short"``.
            entry_price: Entry price.
            quantity: Lot size.

        Returns:
            The created ``PositionInfo``.
        """
        position = PositionInfo(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
        )
        self._positions.append(position)
        self._update_peak_equity()
        return position

    def close_position(self, symbol: str) -> Optional[PositionInfo]:
        """Remove an open position by symbol.

        Args:
            symbol: Trading pair to close.

        Returns:
            The closed ``PositionInfo``, or ``None`` if not found.
        """
        for i, pos in enumerate(self._positions):
            if pos.symbol == symbol:
                return self._positions.pop(i)
        return None

    def set_correlation(
        self,
        symbol_a: str,
        symbol_b: str,
        correlation: float,
    ) -> None:
        """Set the pairwise correlation between two symbols.

        Args:
            symbol_a: First symbol.
            symbol_b: Second symbol.
            correlation: Correlation coefficient (0-1).
        """
        key = tuple(sorted((symbol_a, symbol_b)))
        self._correlation_matrix[key] = correlation

    # ------------------------------------------------------------------
    #  Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self, current_equity: float) -> None:
        """Reset daily P&L tracking at the start of a new trading day.

        Args:
            current_equity: Equity at reset time (becomes the new daily
                starting equity).
        """
        self._daily_start_equity = current_equity
        self._daily_pnl = 0.0

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def peak_equity(self) -> float:
        """Current peak equity."""
        return self._peak_equity

    @property
    def daily_pnl(self) -> float:
        """Current daily P&L."""
        return self._daily_pnl

    @property
    def open_positions(self) -> list[PositionInfo]:
        """List of open positions."""
        return list(self._positions)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _update_peak_equity(self) -> None:
        """Update peak equity if the current equity exceeds the stored peak."""
        # Use the total notional of open positions as a rough equity proxy
        # when no explicit current equity is provided.
        total = sum(
            abs(pos.quantity * pos.entry_price) for pos in self._positions
        )
        if total > self._peak_equity:
            self._peak_equity = total
