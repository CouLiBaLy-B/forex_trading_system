"""Position sizing strategies for forex trading."""

from __future__ import annotations

import math
from typing import Optional

from .models import PositionInfo, RiskMode, RiskParams


class PositionSizer:
    """Calculate position sizes using multiple strategies.

    Supported strategies:
        * **Fixed Fraction** – risk a fixed % of equity per trade.
        * **Kelly Criterion** – optimal bet fraction from win rate / payoff ratio.
        * **ATR-based** – size inversely to instrument volatility.

    All methods honour global min/max lot constraints from *RiskParams*.
    """

    # ── Fixed Fraction ──────────────────────────────────────────────────

    @staticmethod
    def fixed_fraction(
        account_balance: float,
        risk_per_trade_pct: float,
        entry_price: float,
        stop_loss_price: float,
        risk_params: RiskParams,
    ) -> float:
        """Fixed-fraction position sizing.

        Risk amount = account_balance * risk_per_trade_pct.
        Position = risk_amount / (entry_price - stop_loss_price).

        Args:
            account_balance: Current account equity.
            risk_per_trade_pct: Fraction of equity to risk (0-1).
            entry_price: Instrument entry price.
            stop_loss_price: Stop-loss price.
            risk_params: Risk configuration (provides min/max lot sizes).

        Returns:
            Suggested position size in lots, clamped to [min_lot, max_lot].
        """
        risk_amount = account_balance * risk_per_trade_pct
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk <= 0:
            return 0.0

        lots = risk_amount / (price_risk * risk_params.margin_requirement_pct)
        return PositionSizer._clamp(lots, risk_params)

    # ── Kelly Criterion ─────────────────────────────────────────────────

    @staticmethod
    def kelly_criterion(
        win_rate: float,
        payoff_ratio: float,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        risk_params: RiskParams,
    ) -> float:
        """Kelly Criterion position sizing with partial-fraction support.

        Full Kelly = win_rate - (1 - win_rate) / payoff_ratio.
        Uses *risk_params.kelly_fraction* (default 0.25 = quarter-Kelly) to
        temper the bet size and reduce drawdown risk.

        Args:
            win_rate: Historical win rate (0-1).
            payoff_ratio: Average win / average loss.
            account_balance: Current account equity.
            entry_price: Instrument entry price.
            stop_loss_price: Stop-loss price.
            risk_params: Risk configuration (provides min/max lot sizes).

        Returns:
            Suggested position size in lots, clamped to [min_lot, max_lot].
        """
        kelly = win_rate - (1 - win_rate) / payoff_ratio if payoff_ratio > 0 else 0.0
        adjusted = kelly * risk_params.kelly_fraction

        if adjusted <= 0:
            return 0.0

        # Derive a notional stake from the Kelly fraction, then convert to lots.
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk <= 0:
            return 0.0

        lots = (account_balance * adjusted) / (price_risk * risk_params.margin_requirement_pct)
        return PositionSizer._clamp(lots, risk_params)

    # ── ATR-based sizing ────────────────────────────────────────────────

    @staticmethod
    def atr_sizing(
        account_balance: float,
        risk_pct: float,
        atr: float,
        entry_price: float,
        multiplier: float,
        risk_params: RiskParams,
    ) -> float:
        """ATR-based position sizing.

        Position = (account_balance * risk_pct) / (ATR * multiplier).

        This method automatically sets dynamic stop loss at
        ``entry_price ± (atr * multiplier)`` and uses that distance for
        lot calculation.

        Args:
            account_balance: Current account equity.
            risk_pct: Fraction of equity to risk (0-1).
            atr: Average True Range of the instrument.
            entry_price: Instrument entry price.
            multiplier: ATR multiplier for stop distance.
            risk_params: Risk configuration (provides min/max lot sizes).

        Returns:
            Suggested position size in lots, clamped to [min_lot, max_lot].
        """
        risk_amount = account_balance * risk_pct
        stop_distance = atr * multiplier

        if stop_distance <= 0:
            return 0.0

        lots = risk_amount / (stop_distance * risk_params.margin_requirement_pct)
        return PositionSizer._clamp(lots, risk_params)

    # ── Dynamic stop-loss / take-profit ─────────────────────────────────

    @staticmethod
    def calc_dynamic_stop_loss(
        entry_price: float,
        side: str,
        atr: float,
        multiplier: float,
    ) -> float:
        """Calculate a dynamic stop-loss based on ATR.

        For **long** positions the stop sits below entry; for **short**
        positions it sits above.

        Args:
            entry_price: Entry price.
            side: ``"long"`` or ``"short"``.
            atr: Average True Range.
            multiplier: ATR multiplier (typically 1.5-2.5).

        Returns:
            Stop-loss price.
        """
        stop_distance = atr * multiplier
        if side == "long":
            return entry_price - stop_distance
        return entry_price + stop_distance

    @staticmethod
    def calc_dynamic_take_profit(
        entry_price: float,
        side: str,
        atr: float,
        multiplier: float,
        rr_ratio: float = 2.0,
    ) -> float:
        """Calculate a dynamic take-profit based on a fixed risk-reward ratio.

        The distance from entry to TP is ``atr * multiplier * rr_ratio``.

        Args:
            entry_price: Entry price.
            side: ``"long"`` or ``"short"``.
            atr: Average True Range.
            multiplier: ATR multiplier.
            rr_ratio: Risk-reward ratio (default 2.0).

        Returns:
            Take-profit price.
        """
        distance = atr * multiplier * rr_ratio
        if side == "long":
            return entry_price + distance
        return entry_price - distance

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def get_current_position(
        symbol: str,
        open_positions: list[PositionInfo],
    ) -> Optional[PositionInfo]:
        """Return the open PositionInfo for *symbol*, or ``None``."""
        for pos in open_positions:
            if pos.symbol == symbol:
                return pos
        return None

    @staticmethod
    def aggregate_exposure(positions: list[PositionInfo], account_balance: float) -> float:
        """Return total exposure as a fraction of account balance.

        Args:
            positions: List of open positions.
            account_balance: Current account equity.

        Returns:
            Exposure fraction (0-1, may exceed 1 for leveraged accounts).
        """
        if account_balance <= 0:
            return 0.0

        total = sum(
            abs(pos.quantity * pos.entry_price) for pos in positions
        )
        return total / account_balance

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _clamp(lots: float, risk_params: RiskParams) -> float:
        min_lot = getattr(risk_params, "min_lot_size", 0.01)
        max_lot = getattr(risk_params, "max_lot_size", float("inf"))
        return max(min_lot, min(lots, max_lot))
