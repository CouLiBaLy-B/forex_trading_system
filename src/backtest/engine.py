"""Event-driven backtesting engine.

Architecture
============
The engine follows an event-driven pipeline:

1. **Data event** – each candle / tick from the historical data feed.
2. **Signal event** – the strategy emits a trading signal based on current
   indicators and the latest candle data.
3. **Execution event** – the engine simulates order placement, fill
   (with spread + slippage + commission), and updates portfolio state.
4. **Portfolio update** – after every event the portfolio is reconciled:
   positions, cash, equity curve, and drawdown are all refreshed.

This design keeps each concern isolated and makes it straightforward to
swap in different strategies or data sources without touching core logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from src.backtest.models import (
    BacktestConfig,
    BacktestFoldResult,
    BacktestResult,
    BacktestTrade,
    ExitReason,
    PositionSide,
)

logger = logging.getLogger(__name__)

# Type aliases for readability
SignalFn = Callable[[pd.DataFrame, dict[str, Any]], str | None]
"""Strategy function: (indicators_df, state) -> signal or None."""

DataLoaderFn = Callable[..., pd.DataFrame]
"""Data loader callable returning a DataFrame with OHLCV columns."""


# ------ internal data classes (not exposed in __all__) -------------------

@dataclass(slots=True)
class _Order:
    """Internal order representation before fill."""
    instrument: str
    side: PositionSide
    quantity: float
    order_type: str = "MARKET"
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass(slots=True)
class _Position:
    """Open position tracker."""
    instrument: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    commission_paid: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    peak_equity: float = 0.0


# ------ event helper ----------------------------------------------------

@dataclass(slots=True)
class _Event:
    """Base event in the backtest event loop."""
    timestamp: datetime
    index: int
    data: pd.Series


@dataclass(slots=True)
class _SignalEvent(_Event):
    signal: str | None
    strategy_name: str = ""


@dataclass(slots=True)
class _FillEvent(_Event):
    order: _Order
    fill_price: float
    fill_quantity: float
    fill_commission: float


# ------ engine ------------------------------------------------------------

class BacktestEngine:
    """Event-driven backtesting engine.

    Parameters
    ----------
    config :
        Backtest configuration controlling spread, slippage, sizing, etc.
    strategy :
        Callable that receives the indicator DataFrame and a mutable state
        dict and returns a signal string (e.g. ``"BUY"`` / ``"SELL"``) or
        ``None`` for no action.
    data :
        DataFrame with a ``datetime`` index (or column) and OHLCV columns:
        ``open``, ``high``, ``low``, ``close``, ``volume``.
        Optional columns: ``benchmark_close`` for benchmark comparison.
    initial_state :
        Mutable dict passed to the strategy on every call. Useful for
        persisting indicators or custom state across bars.

    Example
    -------
    >>> config = BacktestConfig(instrument="EUR/USD", strategy_name="ma_cross")
    >>> engine = BacktestEngine(config, my_strategy, ohlcv_df)
    >>> result = engine.run()
    >>> print(f"Sharpe: {result.sharpe_ratio:.3f}")
    """

    # ---- column expectations (strict) ----------------------------------

    REQUIRED_OHLCV = {"open", "high", "low", "close", "volume"}
    BENCHMARK_COLS = {"benchmark_close", "benchmark"}

    def __init__(
        self,
        config: BacktestConfig,
        strategy: SignalFn,
        data: pd.DataFrame,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self._raw_data = data.copy()
        self._state: dict[str, Any] = initial_state or {}

        # validate data
        self._validate_data()

        # internal mutable state
        self._position: _Position | None = None
        self._equity_curve: list[float] = []
        self._drawdown_curve: list[float] = []
        self._trade_log: list[BacktestTrade] = []
        self._peak_equity: float = config.initial_capital
        self._cash: float = config.initial_capital
        self._event_index: int = 0
        self._daily_returns: list[float] = []
        self._equity_at_bar_open: float = config.initial_capital
        self._benchmark_curve: list[float] | None = None

        # walking-forward state
        self._fold_index: int = 0

    # ---- public API ----------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest and return the full result.

        This method:
        1. Validates configuration.
        2. Prepares indicator DataFrame (calls ``prepare_indicators``).
        3. Iterates bar-by-bar through the data, emitting events.
        4. Post-processes trades (computes aggregate metrics).
        5. Builds and returns a ``BacktestResult``.

        Returns
        -------
        BacktestResult
            Complete backtesting output with all metrics.
        """
        logger.info(
            "Starting backtest: strategy=%s instrument=%s bars=%d",
            self.config.strategy_name,
            self.config.instrument,
            len(self._raw_data),
        )

        if self._has_walk_forward_config:
            return self._run_walk_forward()

        result = self._run_single()
        logger.info("Backtest complete: total_trades=%d sharpe=%.4f",
                     result.total_trades, result.sharpe_ratio)
        return result

    def run_with_benchmark(self) -> BacktestResult:
        """Run backtest with automatic benchmark comparison.

        If the data contains a benchmark column it is used; otherwise
        a buy-and-hold of the primary instrument is computed as benchmark.

        Returns
        -------
        BacktestResult
        """
        result = self.run()
        return result  # benchmark is computed inside _run_single

    # ---- walking-forward -----------------------------------------------

    def _has_walk_forward_config(self) -> bool:
        return (
            self.config.walk_forward is not None
            and self.config.walk_forward.num_folds > 0
        )

    def _run_walk_forward(self) -> BacktestResult:
        """Execute walking-fold validation.

        Splits data into overlapping in-sample / out-of-sample folds.
        The first fold is in-sample only; subsequent folds alternate.
        Returns the aggregated out-of-sample ``BacktestResult``.
        """
        wf = self.config.walk_forward!
        data_len = len(self._raw_data)
        total_bars = wf.in_sample_period + wf.out_of_sample_period

        if total_bars > data_len:
            raise ValueError(
                f"Data length {data_len} < minimum required {total_bars}"
            )

        fold_results: list[BacktestFoldResult] = []
        all_oot_trades: list[BacktestTrade] = []
        all_oot_equity: list[float] = []
        all_oot_dd: list[float] = []
        all_oot_daily_ret: list[float] = []

        for fold in range(wf.num_folds):
            is_start = fold * wf.step
            is_end = is_start + wf.in_sample_period
            oos_start = is_end
            oos_end = min(oos_start + wf.out_of_sample_period, data_len)

            if oos_end - oos_start <= 0:
                break

            # in-sample: create a temporary engine without the strategy
            # (the strategy was already passed – it is the optimised version)
            temp_state = {**self._state}
            temp_engine = BacktestEngine(
                self.config,
                self.strategy,
                self._raw_data.iloc[is_start:is_end],
                temp_state,
            )
            temp_engine._position = None
            temp_engine._equity_curve = []
            temp_engine._peak_equity = self.config.initial_capital
            temp_engine._cash = self.config.initial_capital
            temp_engine._event_index = 0
            temp_engine._daily_returns = []
            temp_engine._trade_log = []
            temp_engine._drawdown_curve = []
            temp_engine._equity_at_bar_open = self.config.initial_capital
            temp_engine._benchmark_curve = None

            # run in-sample (we keep the trade counts but don't return)
            temp_engine.run()

            # out-of-sample: new engine starting from the end of in-sample
            oos_state = {**temp_state, "best_params": {}}
            oos_engine = BacktestEngine(
                self.config,
                self.strategy,
                self._raw_data.iloc[oos_start:oos_end],
                oos_state,
            )
            oos_engine._equity_curve = list(temp_engine._equity_curve)
            oos_engine._peak_equity = temp_engine._peak_equity
            oos_engine._cash = temp_engine._cash
            oos_engine._equity_at_bar_open = temp_engine._equity_at_bar_open
            oos_result = oos_engine.run()

            fold_result = BacktestFoldResult(
                fold_index=fold,
                in_sample_period=(is_start, is_end),
                out_of_sample_period=(oos_start, oos_end),
                result=oos_result,
                metrics={
                    "sharpe": oos_result.sharpe_ratio,
                    "max_drawdown": oos_result.max_drawdown_pct,
                    "win_rate": oos_result.win_rate,
                    "profit_factor": oos_result.profit_factor,
                },
            )
            fold_results.append(fold_result)

            all_oot_trades.extend(oos_result.trade_log)
            all_oot_equity.extend(oos_result.equity_curve)
            all_oot_dd.extend(oos_result.drawdown_curve)
            all_oot_daily_ret.extend(oos_result.daily_returns)

        # aggregate out-of-sample
        return self._build_aggregate_oos(
            all_oot_trades, all_oot_equity, all_oot_dd,
            all_oot_daily_ret, self._raw_data, fold_results,
        )

    def _build_aggregate_oos(
        self,
        trades: list[BacktestTrade],
        equity: list[float],
        dd: list[float],
        daily_ret: list[float],
        data: pd.DataFrame,
        folds: list[BacktestFoldResult],
    ) -> BacktestResult:
        """Aggregate out-of-sample fold results into a single BacktestResult."""
        n = len(trades)
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl < 0]
        breakeven = [t for t in trades if t.pnl == 0]

        total_win_pnl = sum(t.pnl for t in winning)
        total_lose_pnl = sum(t.pnl for t in losing)

        final_equity = equity[-1] if equity else self.config.initial_capital
        total_ret = (final_equity - self.config.initial_capital) / self.config.initial_capital

        # annualize
        total_days = (data.index[-1] - data.index[0]).days if len(data) > 1 else 1
        ann_ret = total_ret * (365 / max(total_days, 1)) if total_days > 0 else 0.0

        # drawdown stats
        max_dd = max(dd) if dd else 0.0
        avg_dd = np.mean(dd) if dd else 0.0

        # ratios
        sharpe = self._calc_sharpe(daily_ret) if daily_ret else 0.0
        sortino = self._calc_sortino(daily_ret) if daily_ret else 0.0
        omega = self._calc_omega(daily_ret) if daily_ret else 0.0
        pf = total_win_pnl / abs(total_lose_pnl) if total_lose_pnl else 0.0

        # consecutive wins / losses
        cw = self._consecutive_max([1 if t.pnl > 0 else 0 for t in trades])
        cl = self._consecutive_max([1 if t.pnl < 0 else 0 for t in trades])

        return BacktestResult(
            strategy_name=self.config.strategy_name,
            instrument=self.config.instrument,
            start_date=str(data.index[0]),
            end_date=str(data.index[-1]),
            total_trades=n,
            winning_trades=len(winning),
            losing_trades=len(losing),
            breakeven_trades=len(breakeven),
            total_winning_pnl=total_win_pnl,
            total_losing_pnl=total_lose_pnl,
            initial_capital=self.config.initial_capital,
            final_equity=final_equity,
            total_return=total_ret,
            net_return=total_ret - self._total_commission(trades),
            annualized_return=ann_ret,
            total_days=total_days,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd,
            max_drawdown_start="",
            max_drawdown_end="",
            largest_drawdown_day=max(dd) if dd else 0.0,
            avg_drawdown=avg_dd,
            avg_drawdown_pct=avg_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=0.0,
            profit_factor=pf,
            omega_ratio=omega,
            win_rate=len(winning) / max(n, 1),
            avg_win=np.mean([t.pnl for t in winning]) if winning else 0.0,
            avg_loss=np.mean([t.pnl for t in losing]) if losing else 0.0,
            avg_holding_period=np.mean([t.holding_period_seconds for t in trades]) if trades else 0.0,
            avg_holding_bars=np.mean([t.holding_bars for t in trades]) if trades else 0.0,
            largest_win=max(t.pnl for t in winning) if winning else 0.0,
            largest_loss=min(t.pnl for t in losing) if losing else 0.0,
            worst_consecutive_losses=cl,
            best_consecutive_wins=cw,
            total_commission=self._total_commission(trades),
            equity_curve=equity,
            drawdown_curve=dd,
            trade_log=trades,
            daily_returns=daily_ret,
            config=self.config.model_dump(),
            strategy_metadata={"folds": len(folds)},
        )

    # ---- single backtest (non-walking-fold) ----------------------------

    def _run_single(self) -> BacktestResult:
        """Execute one pass through the data."""
        indicators = self._prepare_indicators(self._raw_data)

        # ensure columns exist
        for col in ("close", "open", "high", "low", "volume"):
            if col not in indicators.columns:
                raise ValueError(f"Missing required column: {col}")

        has_benchmark = any(c in indicators.columns for c in self.BENCHMARK_COLS)
        if has_benchmark:
            self._prepare_benchmark(indicators)

        bars = indicators.itertuples(index=True)
        # Skip warm-up bars
        for _ in range(self.config.warm_up_bars):
            bar = next(bars)
            self._on_data_event(bar)

        for bar in bars:
            result = self._on_data_event(bar)
            if result:
                self._process_signal(result)
            self._on_portfolio_update(bar)

        return self._build_result()

    # ---- data preparation ----------------------------------------------

    def _validate_data(self) -> None:
        """Ensure the input DataFrame has the expected structure."""
        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if self._raw_data.empty:
            raise ValueError("Data DataFrame is empty")
        if not hasattr(self._raw_data.index, "dtype") or self._raw_data.index.dtype.kind not in ("M", "O"):
            raise ValueError("Data index must be datetime-like")

    def _prepare_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Call the strategy's indicator preparation if available."""
        # The strategy is a SignalFn, so we expect it to work directly
        # on the OHLCV DataFrame.  No separate preparation is needed
        # unless the strategy itself handles it.
        return data

    def _prepare_benchmark(self, data: pd.DataFrame) -> None:
        """Extract benchmark data if present in the DataFrame."""
        bench_col = next((c for c in self.BENCHMARK_COLS if c in data.columns), None)
        if bench_col is None:
            return
        if bench_col in data.columns:
            self._benchmark_curve = data[bench_col].tolist()

    # ---- event handlers ------------------------------------------------

    def _on_data_event(self, bar: pd.Series) -> _SignalEvent | None:
        """Handle a data event: invoke strategy and return signal event.

        Parameters
        ----------
        bar :
            Current candle as a pandas Series (one row of the indicator DF).

        Returns
        -------
        _SignalEvent | None
            Signal event if the strategy produced a signal, else None.
        """
        signal = self.strategy(self._raw_data, self._state)
        return _SignalEvent(
            timestamp=bar.name if isinstance(bar.name, datetime) else datetime.now(),
            index=self._event_index,
            data=bar,
            signal=signal,
            strategy_name=self.config.strategy_name,
        )

    def _process_signal(self, event: _SignalEvent | None) -> None:
        """Process a signal event: open/close positions, simulate fills."""
        if event is None or event.signal is None:
            return

        # Close existing position if signal differs
        if self._position and event.signal.upper() in ("SELL", "CLOSE"):
            if self._position.side == PositionSide.LONG:
                self._close_position(ExitReason.SIGNAL)

        # Open new long position
        if event.signal.upper() == "BUY" and self._position is None:
            self._open_long(event)
        elif event.signal.upper() == "SELL" and self._position is None:
            self._open_short(event)

    def _open_long(self, event: _SignalEvent) -> None:
        """Open a long position with simulated fill."""
        bar = event.data
        price = self._simulate_fill(bar["close"], spread_side="ask")
        quantity = self._calc_quantity(price)

        if quantity <= 0:
            logger.debug("Empty order: instrument=%s", self.config.instrument)
            return

        commission = self._calc_commission(price * quantity)
        entry_time = event.timestamp if isinstance(event.timestamp, datetime) else datetime.now()

        self._cash -= quantity * price + commission
        self._position = _Position(
            instrument=self.config.instrument,
            side=PositionSide.LONG,
            quantity=quantity,
            entry_price=price,
            entry_time=entry_time,
            commission_paid=commission,
        )
        self._event_index += 1

        logger.debug(
            "OPEN LONG: %.6f @ %.6f qty=%.4f comm=%.4f",
            price, bar.get("open", 0), quantity, commission,
        )

    def _open_short(self, event: _SignalEvent) -> None:
        """Open a short position with simulated fill."""
        bar = event.data
        price = self._simulate_fill(bar["close"], spread_side="bid")
        quantity = self._calc_quantity(price)

        if quantity <= 0:
            logger.debug("Empty order: instrument=%s", self.config.instrument)
            return

        commission = self._calc_commission(price * quantity)
        entry_time = event.timestamp if isinstance(event.timestamp, datetime) else datetime.now()

        self._cash += quantity * price - commission
        self._position = _Position(
            instrument=self.config.instrument,
            side=PositionSide.SHORT,
            quantity=quantity,
            entry_price=price,
            entry_time=entry_time,
            commission_paid=commission,
        )
        self._event_index += 1

        logger.debug(
            "OPEN SHORT: %.6f @ %.6f qty=%.4f comm=%.4f",
            price, bar.get("open", 0), quantity, commission,
        )

    def _close_position(self, reason: ExitReason) -> BacktestTrade | None:
        """Close the current position and record the trade."""
        if self._position is None:
            return None

        bar = self._raw_data.iloc[min(self._event_index, len(self._raw_data) - 1)] if hasattr(self._raw_data, 'iloc') else None
        bar_close = bar["close"] if bar is not None else self._position.entry_price

        if self._position.side == PositionSide.LONG:
            exit_price = self._simulate_fill(bar_close, spread_side="bid")
            pnl = (exit_price - self._position.entry_price) * self._position.quantity - self._position.commission_paid
            self._cash += self._position.quantity * exit_price - self._position.commission_paid
        else:
            exit_price = self._simulate_fill(bar_close, spread_side="ask")
            pnl = (self._position.entry_price - exit_price) * self._position.quantity - self._position.commission_paid
            self._cash -= self._position.quantity * exit_price - self._position.commission_paid

        exit_time = datetime.now()

        trade = BacktestTrade(
            instrument=self.config.instrument,
            side=self._position.side,
            entry_price=self._position.entry_price,
            exit_price=exit_price,
            quantity=self._position.quantity,
            entry_time=self._position.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_pct=(pnl / (self._position.entry_price * self._position.quantity)) * 100
                    if self._position.entry_price > 0 else 0.0,
            commission=self._position.commission_paid,
            exit_reason=reason,
            holding_bars=self._event_index,
            strategy_name=self.config.strategy_name,
        )
        self._trade_log.append(trade)

        logger.debug(
            "CLOSE: side=%s reason=%s pnl=%.4f",
            self._position.side, reason.value, pnl,
        )

        self._position = None
        return trade

    # ---- portfolio update ----------------------------------------------

    def _on_portfolio_update(self, bar: pd.Series) -> None:
        """Update equity curve and drawdown after every bar."""
        if self._position:
            bar_close = bar["close"]
            if self._position.side == PositionSide.LONG:
                unrealized = (bar_close - self._position.entry_price) * self._position.quantity
            else:
                unrealized = (self._position.entry_price - bar_close) * self._position.quantity
            current_equity = self._cash + unrealized
        else:
            current_equity = self._cash

        self._peak_equity = max(self._peak_equity, current_equity)
        self._equity_curve.append(current_equity)
        dd = (self._peak_equity - current_equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        self._drawdown_curve.append(dd)

        # daily return tracking
        if self._equity_at_bar_open > 0:
            ret = (current_equity - self._equity_at_bar_open) / self._equity_at_bar_open
            self._daily_returns.append(ret)
        self._equity_at_bar_open = current_equity

    # ---- fill simulation -----------------------------------------------

    def _simulate_fill(self, price: float, spread_side: str) -> float:
        """Simulate an order fill with spread + slippage.

        Parameters
        ----------
        price :
            Base price from the candle (close by default).
        spread_side :
            "ask" for buy fills (price increases), "bid" for sell fills
            (price decreases).
        """
        half_spread = self.config.spread / 2
        if spread_side == "ask":
            fill = price + half_spread + self.config.slippage
        else:
            fill = price - half_spread - self.config.slippage
        return fill

    def _calc_quantity(self, price: float) -> float:
        """Calculate position size based on config risk parameters."""
        equity = self._peak_equity if self._peak_equity > 0 else self.config.initial_capital
        dollar_risk = equity * self.config.risk_per_trade
        max_dollar = equity * self.config.max_position_size

        quantity = min(dollar_risk, max_dollar) / price if price > 0 else 0
        quantity = min(quantity, equity * self.config.leverage / price) if price > 0 else 0
        return max(quantity, 0)

    def _calc_commission(self, notional: float) -> float:
        """Calculate commission based on config mode."""
        if self.config.commission_mode == "per_unit":
            qty = notional / max(self._position.entry_price, 1e-12) if self._position else 1
            return qty * self.config.commission_per_unit
        return notional * self.config.commission_rate

    # ---- result builder ------------------------------------------------

    def _build_result(self) -> BacktestResult:
        """Compile all accumulated data into a BacktestResult."""
        trades = self._trade_log
        equity = self._equity_curve
        dd_curve = self._drawdown_curve
        daily_ret = self._daily_returns

        n = len(trades)
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl < 0]
        breakeven = [t for t in trades if t.pnl == 0]

        total_win_pnl = sum(t.pnl for t in winning)
        total_lose_pnl = sum(t.pnl for t in losing)

        final_equity = equity[-1] if equity else self.config.initial_capital
        total_ret = (final_equity - self.config.initial_capital) / self.config.initial_capital

        total_days = len(equity)
        ann_ret = total_ret * (self.config.trading_days_per_year / total_days) if total_days > 0 else 0.0

        max_dd = max(dd_curve) if dd_curve else 0.0
        avg_dd = np.mean(dd_curve) if dd_curve else 0.0

        sharpe = self._calc_sharpe(daily_ret) if daily_ret else 0.0
        sortino = self._calc_sortino(daily_ret) if daily_ret else 0.0
        omega = self._calc_omega(daily_ret) if daily_ret else 0.0

        gross_profit = total_win_pnl
        gross_loss = abs(total_lose_pnl) if total_lose_pnl else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit > 0 else 0.0)

        cw = self._consecutive_max([1 if t.pnl > 0 else 0 for t in trades])
        cl = self._consecutive_max([1 if t.pnl < 0 else 0 for t in trades])

        total_comm = self._total_commission(trades)

        # benchmark
        bench_ret = 0.0
        bench_sharpe = 0.0
        if self._benchmark_curve:
            bench_ret = (self._benchmark_curve[-1] - self._benchmark_curve[0]) / self._benchmark_curve[0]
            bench_sharpe = self._calc_sharpe(
                [
                    (self._benchmark_curve[i] - self._benchmark_curve[i - 1])
                    / self._benchmark_curve[i - 1]
                    for i in range(1, len(self._benchmark_curve))
                ]
            )

        return BacktestResult(
            strategy_name=self.config.strategy_name,
            instrument=self.config.instrument,
            start_date=str(self._raw_data.index[0]) if len(self._raw_data) else "",
            end_date=str(self._raw_data.index[-1]) if len(self._raw_data) else "",
            total_trades=n,
            winning_trades=len(winning),
            losing_trades=len(losing),
            breakeven_trades=len(breakeven),
            total_winning_pnl=total_win_pnl,
            total_losing_pnl=total_lose_pnl,
            initial_capital=self.config.initial_capital,
            final_equity=final_equity,
            total_return=total_ret,
            net_return=total_ret - (total_comm / self.config.initial_capital),
            annualized_return=ann_ret,
            total_days=total_days,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd,
            max_drawdown_start="",
            max_drawdown_end="",
            largest_drawdown_day=max_dd,
            avg_drawdown=avg_dd,
            avg_drawdown_pct=avg_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=0.0,
            profit_factor=pf,
            omega_ratio=omega,
            win_rate=len(winning) / max(n, 1),
            avg_win=np.mean([t.pnl for t in winning]) if winning else 0.0,
            avg_loss=np.mean([t.pnl for t in losing]) if losing else 0.0,
            avg_holding_period=np.mean([t.holding_period_seconds for t in trades]) if trades else 0.0,
            avg_holding_bars=np.mean([t.holding_bars for t in trades]) if trades else 0.0,
            largest_win=max(t.pnl for t in winning) if winning else 0.0,
            largest_loss=min(t.pnl for t in losing) if losing else 0.0,
            worst_consecutive_losses=cl,
            best_consecutive_wins=cw,
            total_commission=total_comm,
            equity_curve=equity,
            drawdown_curve=dd_curve,
            trade_log=trades,
            daily_returns=daily_ret,
            benchmark_return=bench_ret,
            benchmark_sharpe=bench_sharpe,
            excess_return=total_ret - bench_ret,
            config=self.config.model_dump(),
        )

    # ---- helper methods ------------------------------------------------

    @staticmethod
    def _calc_sharpe(returns: list[float]) -> float:
        """Annualised Sharpe ratio (risk-free = 0)."""
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        mean = arr.mean()
        std = arr.std(ddof=1)
        if std == 0:
            return 0.0
        return (mean / std) * (252 ** 0.5)

    @staticmethod
    def _calc_sortino(returns: list[float]) -> float:
        """Annualised Sortino ratio."""
        if not returns:
            return 0.0
        arr = np.array(returns)
        downside = arr[arr < 0]
        if len(downside) == 0:
            return float("inf") if arr.mean() > 0 else 0.0
        ds = downside.std(ddof=1)
        if ds == 0:
            return 0.0
        return (arr.mean() / ds) * (252 ** 0.5)

    @staticmethod
    def _calc_omega(returns: list[float], threshold: float = 0.0) -> float:
        """Omega ratio: proportion of returns above threshold / below."""
        if not returns:
            return 0.0
        gains = sum(max(r - threshold, 0) for r in returns)
        losses = sum(max(threshold - r, 0) for r in returns)
        return gains / losses if losses > 0 else float("inf") if gains > 0 else 0.0

    @staticmethod
    def _consecutive_max(flag_list: list[int]) -> int:
        """Maximum run of 1s in a binary list."""
        if not flag_list:
            return 0
        mx = cur = 0
        for f in flag_list:
            if f:
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 0
        return mx

    @staticmethod
    def _total_commission(trades: list[BacktestTrade]) -> float:
        """Sum of all commissions across trades."""
        return sum(t.commission for t in trades)
