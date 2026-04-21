"""PerformanceTracker – compute trading performance metrics.

Calculates Sharpe ratio, Sortino ratio, max drawdown, win rate,
profit factor, Calmar ratio, rolling metrics, and equity curves
from a portfolio's trade history.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .manager import PortfolioManager
from .models import PerformanceMetrics, PositionSide, RollingMetrics, TradeRecord


class PerformanceTracker:
    """Compute and cache performance metrics from a PortfolioManager.

    Provides industry-standard risk/return metrics calculated from the
    trade history and equity curve of a portfolio.

    Attributes:
        _metrics: Cached PerformanceMetrics (None until computed).
        _equity_curve: Cached equity curve as a pandas Series (index=timestamp).
        _rolling: Cached dict of rolling metric windows.
    """

    def __init__(self, portfolio: PortfolioManager) -> None:
        """Initialize with a PortfolioManager instance.

        Args:
            portfolio: PortfolioManager whose trades/metrics will be analyzed.
        """
        self._portfolio = portfolio
        self._metrics: Optional[PerformanceMetrics] = None
        self._equity_curve: Optional[pd.Series] = None
        self._rolling: dict[str, RollingMetrics] = {}

    # ------------------------------------------------------------------
    #  Public API – compute metrics
    # ------------------------------------------------------------------

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Compute and return all standard performance metrics.

        Calls the underlying computation methods and caches the result.

        Returns:
            PerformanceMetrics with all computed ratios.
        """
        if self._metrics is None:
            self._metrics = self._compute_all_metrics()
        return self._metrics

    def get_equity_curve(self) -> pd.Series:
        """Compute the equity curve from trade history.

        The equity curve records cumulative P&L at each trade's exit
        timestamp, plus the starting equity.

        Returns:
            pandas Series indexed by datetime with equity values.
        """
        if self._equity_curve is None:
            self._equity_curve = self._compute_equity_curve()
        return self._equity_curve

    def get_rolling_metrics(
        self,
        window: timedelta = timedelta(days=30),
    ) -> dict[str, RollingMetrics]:
        """Compute rolling performance metrics over fixed-size windows.

        Windows are determined by the most recent trade's exit time.

        Args:
            window: Size of the rolling window (e.g. 30 days).

        Returns:
            Mapping of window start (ISO string) -> RollingMetrics.
        """
        trades = self._portfolio.get_trade_history()
        if not trades:
            return {}

        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        latest = sorted_trades[-1].exit_time
        results: dict[str, RollingMetrics] = {}

        # Slide window backward in chunks of window / 2
        step = window / 2
        t = latest
        while t > sorted_trades[0].entry_time:
            win_start = t - window
            win_end = t
            window_trades = [
                tr for tr in sorted_trades
                if win_start <= tr.exit_time <= win_end
            ]
            if window_trades:
                start_str = win_start.isoformat()
                rolling = RollingMetrics(
                    window_start=win_start,
                    window_end=win_end,
                    sharpe_ratio=self._sharpe_ratio_from_trades(window_trades),
                    max_drawdown=self._max_drawdown_from_trades(window_trades),
                    win_rate=self._win_rate_from_trades(window_trades),
                    total_pnl=sum(t_.realized_pnl for t_ in window_trades),
                    total_return=self._total_return_from_trades(window_trades),
                    total_trades=len(window_trades),
                )
                results[start_str] = rolling
            t -= step

        self._rolling = results
        return results

    # ------------------------------------------------------------------
    #  Core metrics
    # ------------------------------------------------------------------

    def _compute_all_metrics(self) -> PerformanceMetrics:
        """Compute every performance metric from the trade history.

        Returns:
            Fully populated PerformanceMetrics DTO.
        """
        trades = self._portfolio.get_trade_history()
        if not trades:
            return PerformanceMetrics()

        pnl_values = [t.realized_pnl for t in trades]
        total_pnl = sum(pnl_values)

        winning = [t for t in trades if t.realized_pnl > 0]
        losing = [t for t in trades if t.realized_pnl <= 0]
        total_trades = len(trades)
        winning_count = len(winning)
        losing_count = len(losing)

        gross_wins = sum(t.realized_pnl for t in winning) if winning else 0.0
        gross_losses = abs(sum(t.realized_pnl for t in losing)) if losing else 0.0

        # Returns as daily series for Sharpe/Sortino
        daily_returns = self._compute_daily_returns(trades)

        # Sharpe ratio (risk-free rate = 0, annualized)
        sharpe = self._sharpe_ratio(daily_returns)

        # Sortino ratio (risk-free rate = 0, annualized)
        sortino = self._sortino_ratio(daily_returns)

        # Max drawdown
        equity_curve = self.get_equity_curve()
        max_dd = self._max_drawdown_series(equity_curve)

        # Calmar ratio
        total_return = self._total_return_from_trades(trades)
        calmar = total_return / max_dd if max_dd > 0 else 0.0

        # Average holding period
        holding_days = [
            (t.exit_time - t.entry_time).total_seconds() / 86400.0
            for t in trades
        ]
        avg_holding = sum(holding_days) / len(holding_days) if holding_days else 0.0

        # Current drawdown from portfolio state
        state = self._portfolio.get_portfolio_state()
        current_dd = state.max_drawdown

        return PerformanceMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration_days=self._max_drawdown_duration(equity_curve),
            win_rate=winning_count / total_trades if total_trades else 0.0,
            profit_factor=gross_wins / gross_losses if gross_losses > 0 else gross_wins,
            avg_win=gross_wins / winning_count if winning_count else 0.0,
            avg_loss=(gross_losses / losing_count if losing_count else 0.0) * -1,
            calmar_ratio=calmar,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            avg_holding_period_days=avg_holding,
            total_pnl=total_pnl,
            current_drawdown=current_dd,
        )

    # ------------------------------------------------------------------
    #  Sharpe ratio – daily returns, risk-free rate = 0
    # ------------------------------------------------------------------

    def _sharpe_ratio(self, daily_returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio (risk-free rate = 0).

        Sharpe = (mean(daily_returns) / std(daily_returns)) * sqrt(252)

        Args:
            daily_returns: Series of daily returns.

        Returns:
            Annualized Sharpe ratio.
        """
        if len(daily_returns) < 2:
            return 0.0
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        if std_ret == 0:
            return 0.0
        return (mean_ret / std_ret) * math.sqrt(252)

    def _sharpe_ratio_from_trades(self, trades: list[TradeRecord]) -> float:
        """Calculate Sharpe ratio directly from a list of trades.

        Args:
            trades: Subset of trades to analyze.

        Returns:
            Sharpe ratio as a float.
        """
        if len(trades) < 2:
            return 0.0
        pnls = np.array([t.realized_pnl for t in trades], dtype=np.float64)
        mean_pnl = pnls.mean()
        std_pnl = pnls.std()
        if std_pnl == 0:
            return 0.0
        return float((mean_pnl / std_pnl) * math.sqrt(252))

    # ------------------------------------------------------------------
    #  Sortino ratio – downside deviation, risk-free rate = 0
    # ------------------------------------------------------------------

    def _sortino_ratio(self, daily_returns: pd.Series) -> float:
        """Calculate annualized Sortino ratio (risk-free rate = 0).

        Sortino = (mean(daily_returns) / downside_deviation) * sqrt(252)
        Only negative returns contribute to downside deviation.

        Args:
            daily_returns: Series of daily returns.

        Returns:
            Annualized Sortino ratio.
        """
        if len(daily_returns) < 2:
            return 0.0
        mean_ret = daily_returns.mean()
        downside = daily_returns[daily_returns < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = downside.std()
        if downside_std == 0:
            return 0.0
        return (mean_ret / downside_std) * math.sqrt(252)

    # ------------------------------------------------------------------
    #  Max drawdown
    # ------------------------------------------------------------------

    def _max_drawdown_series(self, equity_curve: pd.Series) -> float:
        """Compute maximum drawdown from peak equity.

        Drawdown at each point = 1 - (equity / peak_equity).
        Max drawdown is the largest such value.

        Args:
            equity_curve: Series of equity values indexed by time.

        Returns:
            Maximum drawdown as a fraction (0-1).
        """
        if equity_curve.empty:
            return 0.0
        peak = equity_curve.cummax()
        drawdowns = 1.0 - (equity_curve / peak)
        return float(drawdowns.max())

    def _max_drawdown_from_trades(self, trades: list[TradeRecord]) -> float:
        """Compute max drawdown from a list of trade P&Ls.

        Args:
            trades: Subset of trades.

        Returns:
            Maximum drawdown as a fraction (0-1).
        """
        if not trades:
            return 0.0
        equity = [0.0]
        for t in sorted(trades, key=lambda x: x.exit_time):
            equity.append(equity[-1] + t.realized_pnl)
        return self._max_drawdown_series(pd.Series(equity))

    def _max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Compute the longest consecutive drawdown period in days.

        Args:
            equity_curve: Series of equity values indexed by time.

        Returns:
            Maximum consecutive days in drawdown.
        """
        if equity_curve.empty:
            return 0
        peak = equity_curve.cummax()
        in_dd = equity_curve < peak
        if not in_dd.any():
            return 0
        # Count longest run of True values
        max_duration = 0
        current_duration = 0
        for val in in_dd:
            if val:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        return max_duration

    # ------------------------------------------------------------------
    #  Win rate
    # ------------------------------------------------------------------

    def _win_rate_from_trades(self, trades: list[TradeRecord]) -> float:
        """Calculate win rate from a list of trades.

        Win rate = winning_trades / total_trades.

        Args:
            trades: Subset of trades to analyze.

        Returns:
            Win rate as a fraction (0-1).
        """
        if not trades:
            return 0.0
        winning = sum(1 for t in trades if t.realized_pnl > 0)
        return winning / len(trades)

    # ------------------------------------------------------------------
    #  Profit factor – gross wins / gross losses
    # ------------------------------------------------------------------

    def _total_return_from_trades(self, trades: list[TradeRecord]) -> float:
        """Calculate total return as a fraction of initial equity.

        Args:
            trades: All trades in the portfolio.

        Returns:
            Total return fraction (e.g. 0.15 = 15%).
        """
        if not trades:
            return 0.0
        total_pnl = sum(t.realized_pnl for t in trades)
        initial_equity = self._portfolio.peak_equity
        return total_pnl / initial_equity if initial_equity > 0 else 0.0

    # ------------------------------------------------------------------
    #  Daily returns
    # ------------------------------------------------------------------

    def _compute_daily_returns(self, trades: list[TradeRecord]) -> pd.Series:
        """Build a daily returns time series from trade P&L values.

        For each trading day where at least one trade closed, the return
        is computed as P&L / initial_equity. Days with no trades get 0.0.

        Args:
            trades: All trade records.

        Returns:
            pandas Series of daily returns indexed by date.
        """
        if not trades:
            return pd.Series(dtype=float)

        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        daily_pnl: dict[str, float] = {}
        for t in sorted_trades:
            day = t.exit_time.date().isoformat()
            daily_pnl[day] = daily_pnl.get(day, 0.0) + t.realized_pnl

        initial_equity = self._portfolio.peak_equity
        daily_returns = {
            day: pnl / initial_equity if initial_equity > 0 else 0.0
            for day, pnl in daily_pnl.items()
        }
        return pd.Series(daily_returns, dtype=float)

    # ------------------------------------------------------------------
    #  Equity curve
    # ------------------------------------------------------------------

    def _compute_equity_curve(self) -> pd.Series:
        """Build the equity curve from the portfolio's trade history.

        The curve starts at initial equity and steps through each trade's
        exit, adding the realized P&L.

        Returns:
            pandas Series indexed by exit_time with cumulative equity values.
        """
        trades = self._portfolio.get_trade_history()
        if not trades:
            initial = self._portfolio.peak_equity
            now = datetime.now(timezone.utc)
            return pd.Series([initial], index=[now])

        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        initial_equity = self._portfolio.peak_equity
        cumulative = [initial_equity]
        index: list[datetime] = [sorted_trades[0].exit_time]

        for t in sorted_trades:
            cumulative.append(cumulative[-1] + t.realized_pnl)
            index.append(t.exit_time)

        return pd.Series(cumulative, index=pd.DatetimeIndex(index))
