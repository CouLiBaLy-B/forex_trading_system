"""Reporting utilities for backtest results.

Provides ``BacktestReporter`` to export results as CSV, JSON summary,
equity curve data, trade log, and benchmark comparison.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.backtest.models import BacktestResult, BacktestTrade

logger = logging.getLogger(__name__)


class BacktestReporter:
    """Generate reports from a ``BacktestResult``.

    Parameters
    ----------
    result :
        Complete backtesting output from ``BacktestEngine.run()``.
    output_dir :
        Directory path where generated files are written.  Defaults to the
        current working directory.

    Example
    -------
    >>> result = engine.run()
    >>> reporter = BacktestReporter(result, output_dir="./reports")
    >>> reporter.generate_all()
    """

    def __init__(self, result: BacktestResult, output_dir: str | Path = ".") -> None:
        self.result = result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- public API ------------------------------------------------

    def generate_all(self) -> dict[str, Path]:
        """Generate every report type and return paths.

        Returns
        -------
        dict
            Mapping of report name to output file path.
        """
        outputs = {
            "csv": self.generate_csv_report(),
            "json": self.generate_json_summary(),
            "equity": self.generate_equity_curve(),
            "trade_log": self.generate_trade_log(),
            "benchmark": self.generate_benchmark_comparison(),
        }
        logger.info("Generated %d reports in %s", len(outputs), self.output_dir)
        return outputs

    # ---- CSV report ------------------------------------------------

    def generate_csv_report(self, path: str | Path | None = None) -> Path:
        """Write the full performance metrics table to CSV.

        Parameters
        ----------
        path :
            Optional output path.  Defaults to
            ``<output_dir>/backtest_report.csv``.

        Returns
        -------
        Path
            Absolute path of the generated file.
        """
        out = Path(path) if path else self.output_dir / "backtest_report.csv"
        rows: list[tuple[str, str]] = [
            ("Strategy", self.result.strategy_name),
            ("Instrument", self.result.instrument),
            ("Period", f"{self.result.start_date} -> {self.result.end_date}"),
            ("Initial Capital", f"{self.result.initial_capital:,.2f}"),
            ("Final Equity", f"{self.result.final_equity:,.2f}"),
            ("Total Return", f"{self.result.total_return:.6f}"),
            ("Net Return", f"{self.result.net_return:.6f}"),
            ("Annualized Return", f"{self.result.annualized_return:.6f}"),
            ("", ""),
            ("Performance", ""),
            ("Total Trades", str(self.result.total_trades)),
            ("Winning Trades", str(self.result.winning_trades)),
            ("Losing Trades", str(self.result.losing_trades)),
            ("Breakeven Trades", str(self.result.breakeven_trades)),
            ("Win Rate", f"{self.result.win_rate:.4f}"),
            ("Profit Factor", f"{self.result.profit_factor:.4f}"),
            ("Gross Profit", f"{self.result.total_winning_pnl:,.2f}"),
            ("Gross Loss", f"{self.result.total_losing_pnl:,.2f}"),
            ("Largest Win", f"{self.result.largest_win:,.2f}"),
            ("Largest Loss", f"{self.result.largest_loss:,.2f}"),
            ("Avg Win", f"{self.result.avg_win:,.2f}"),
            ("Avg Loss", f"{self.result.avg_loss:,.2f}"),
            ("Avg Holding Bars", f"{self.result.avg_holding_bars:.1f}"),
            ("Avg Holding Period (s)", f"{self.result.avg_holding_period:.1f}"),
            ("", ""),
            ("Risk", ""),
            ("Max Drawdown", f"{self.result.max_drawdown_pct:.6f}"),
            ("Max DD Start", self.result.max_drawdown_start),
            ("Max DD End", self.result.max_drawdown_end),
            ("Avg Drawdown", f"{self.result.avg_drawdown_pct:.6f}"),
            ("Worst Consecutive Losses", str(self.result.worst_consecutive_losses)),
            ("Best Consecutive Wins", str(self.result.best_consecutive_wins)),
            ("", ""),
            ("Ratios", ""),
            ("Sharpe Ratio", f"{self.result.sharpe_ratio:.4f}"),
            ("Sortino Ratio", f"{self.result.sortino_ratio:.4f}"),
            ("Calmar Ratio", f"{self.result.calmar_computed:.4f}"),
            ("Omega Ratio", f"{self.result.omega_ratio:.4f}"),
            ("P/L Ratio", f"{self.result.profit_loss_ratio:.4f}"),
            ("", ""),
            ("Costs", ""),
            ("Total Commission", f"{self.result.total_commission:,.4f}"),
            ("Total Slippage Cost", f"{self.result.total_slippage_cost:,.4f}"),
        ]

        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Metric", "Value"])
            writer.writerows(rows)

        logger.info("CSV report written to %s", out)
        return out

    # ---- JSON summary -------------------------------------------

    def generate_json_summary(self, path: str | Path | None = None) -> Path:
        """Write a JSON summary of the backtest result.

        The output is serialisable (no pandas / numpy objects).

        Parameters
        ----------
        path :
            Optional output path.  Defaults to
            ``<output_dir>/backtest_summary.json``.

        Returns
        -------
        Path
        """
        out = Path(path) if path else self.output_dir / "backtest_summary.json"

        summary = {
            "strategy_name": self.result.strategy_name,
            "instrument": self.result.instrument,
            "period": {
                "start": self.result.start_date,
                "end": self.result.end_date,
            },
            "trades": {
                "total": self.result.total_trades,
                "winning": self.result.winning_trades,
                "losing": self.result.losing_trades,
                "breakeven": self.result.breakeven_trades,
                "win_rate": round(self.result.win_rate, 6),
            },
            "returns": {
                "total": round(self.result.total_return, 6),
                "net": round(self.result.net_return, 6),
                "annualized": round(self.result.annualized_return, 6),
            },
            "risk": {
                "max_drawdown": round(self.result.max_drawdown_pct, 6),
                "max_drawdown_start": self.result.max_drawdown_start,
                "max_drawdown_end": self.result.max_drawdown_end,
                "avg_drawdown": round(self.result.avg_drawdown_pct, 6),
            },
            "ratios": {
                "sharpe": round(self.result.sharpe_ratio, 4),
                "sortino": round(self.result.sortino_ratio, 4),
                "calmar": round(self.result.calmar_computed, 4),
                "omega": round(self.result.omega_ratio, 4),
                "profit_factor": round(self.result.profit_factor, 4),
                "profit_loss_ratio": round(self.result.profit_loss_ratio, 4),
            },
            "trade_statistics": {
                "avg_win": round(self.result.avg_win, 4),
                "avg_loss": round(self.result.avg_loss, 4),
                "largest_win": round(self.result.largest_win, 4),
                "largest_loss": round(self.result.largest_loss, 4),
                "avg_holding_bars": round(self.result.avg_holding_bars, 2),
                "best_consecutive_wins": self.result.best_consecutive_wins,
                "worst_consecutive_losses": self.result.worst_consecutive_losses,
            },
            "costs": {
                "total_commission": round(self.result.total_commission, 4),
                "total_slippage_cost": round(self.result.total_slippage_cost, 4),
            },
            "benchmark": {
                "return": round(self.result.benchmark_return, 6),
                "sharpe": round(self.result.benchmark_sharpe, 4),
                "excess_return": round(self.result.excess_return, 6),
            },
            "initial_capital": self.result.initial_capital,
            "final_equity": round(self.result.final_equity, 4),
            "total_pnl": round(self.result.total_pnl, 4),
        }

        with out.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        logger.info("JSON summary written to %s", out)
        return out

    # ---- equity curve ------------------------------------

    def generate_equity_curve(
        self, path: str | Path | None = None, with_benchmark: bool = True,
    ) -> Path:
        """Generate equity curve CSV (index, equity[, benchmark]).

        Parameters
        ----------
        path :
            Optional output path.  Defaults to
            ``<output_dir>/equity_curve.csv``.
        with_benchmark :
            Include a benchmark column if available.

        Returns
        -------
        Path
        """
        out = Path(path) if path else self.output_dir / "equity_curve.csv"

        equity = self.result.equity_curve
        data_rows: list[list[str]] = []
        for i, eq in enumerate(equity):
            row = [str(i), f"{eq:.4f}"]
            if with_benchmark and self.result.benchmark_curve and i < len(self.result.benchmark_curve):
                row.append(f"{self.result.benchmark_curve[i]:.4f}")
            data_rows.append(row)

        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            header = ["index", "equity"]
            if with_benchmark and self.result.benchmark_curve:
                header.append("benchmark")
            writer.writerow(header)
            writer.writerows(data_rows)

        logger.info("Equity curve written to %s (%d bars)", out, len(equity))
        return out

    # ---- trade log ------------------------------------

    def generate_trade_log(
        self, path: str | Path | None = None,
    ) -> Path:
        """Write individual trade records to CSV.

        Parameters
        ----------
        path :
            Optional output path.  Defaults to
            ``<output_dir>/trade_log.csv``.

        Returns
        -------
        Path
        """
        out = Path(path) if path else self.output_dir / "trade_log.csv"

        columns = [
            "trade_id", "instrument", "side", "entry_time", "exit_time",
            "entry_price", "exit_price", "quantity", "pnl", "pnl_pct",
            "commission", "holding_bars", "holding_seconds", "exit_reason",
            "strategy",
        ]

        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for i, trade in enumerate(self.result.trade_log):
                writer.writerow([
                    i + 1,
                    trade.instrument,
                    trade.side.value,
                    str(trade.entry_time) if trade.entry_time else "",
                    str(trade.exit_time) if trade.exit_time else "",
                    f"{trade.entry_price:.6f}",
                    f"{trade.exit_price:.6f}",
                    f"{trade.quantity:.4f}",
                    f"{trade.pnl:.4f}",
                    f"{trade.pnl_pct:.4f}",
                    f"{trade.commission:.4f}",
                    trade.holding_bars,
                    trade.holding_period_seconds,
                    trade.exit_reason.value,
                    trade.strategy_name,
                ])

        logger.info("Trade log written to %s (%d trades)", out, len(self.result.trade_log))
        return out

    # ---- benchmark comparison ------------------------------

    def generate_benchmark_comparison(
        self, path: str | Path | None = None,
    ) -> Path:
        """Generate benchmark comparison CSV with side-by-side returns.

        Columns: index, strategy_return, benchmark_return, cumulative_strategy,
                 cumulative_benchmark, excess_return.

        Parameters
        ----------
        path :
            Optional output path.  Defaults to
            ``<output_dir>/benchmark_comparison.csv``.

        Returns
        -------
        Path
        """
        out = Path(path) if path else self.output_dir / "benchmark_comparison.csv"

        n = len(self.result.equity_curve)
        strategy_curve = self.result.equity_curve
        bench_curve = (
            self.result.benchmark_curve
            if self.result.benchmark_curve
            else [0.0] * n
        )

        # compute daily returns
        strat_ret = []
        for i in range(1, n):
            if strategy_curve[i - 1] > 0:
                strat_ret.append(
                    (strategy_curve[i] - strategy_curve[i - 1]) / strategy_curve[i - 1]
                )
            else:
                strat_ret.append(0.0)

        bench_ret = []
        for i in range(1, n):
            if bench_curve[i - 1] > 0:
                bench_ret.append(
                    (bench_curve[i] - bench_curve[i - 1]) / bench_curve[i - 1]
                )
            else:
                bench_ret.append(0.0)

        # cumulative
        cum_strategy = np.cumprod([1.0] + [1.0 + r for r in strat_ret]) - 1.0
        cum_bench = np.cumprod([1.0] + [1.0 + r for r in bench_ret]) - 1.0
        excess = cum_strategy - cum_bench

        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "index",
                "strategy_return",
                "benchmark_return",
                "cumulative_strategy",
                "cumulative_benchmark",
                "excess_return",
            ])
            for i in range(n):
                sr = strat_ret[i] if i < len(strat_ret) else 0.0
                br = bench_ret[i] if i < len(bench_ret) else 0.0
                writer.writerow([
                    i,
                    f"{sr:.6f}",
                    f"{br:.6f}",
                    f"{cum_strategy[i]:.6f}",
                    f"{cum_bench[i]:.6f}",
                    f"{excess[i]:.6f}",
                ])

        logger.info("Benchmark comparison written to %s (%d bars)", out, n)
        return out

    # ---- in-memory helpers -----------------------------------

    def to_json_str(self) -> str:
        """Return JSON summary as a string (no file write)."""
        return json.dumps({
            "strategy_name": self.result.strategy_name,
            "instrument": self.result.instrument,
            "total_trades": self.result.total_trades,
            "win_rate": self.result.win_rate,
            "total_return": self.result.total_return,
            "sharpe_ratio": self.result.sharpe_ratio,
            "sortino_ratio": self.result.sortino_ratio,
            "calmar_ratio": self.result.calmar_computed,
            "max_drawdown": self.result.max_drawdown_pct,
            "profit_factor": self.result.profit_factor,
            "final_equity": self.result.final_equity,
        }, default=str)

    def to_csv_str(self) -> str:
        """Return full metrics as a CSV string (no file write)."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Metric", "Value"])
        writer.writerow(("Strategy", self.result.strategy_name))
        writer.writerow(("Instrument", self.result.instrument))
        writer.writerow(("Total Return", f"{self.result.total_return:.6f}"))
        writer.writerow(("Sharpe Ratio", f"{self.result.sharpe_ratio:.4f}"))
        writer.writerow(("Sortino Ratio", f"{self.result.sortino_ratio:.4f}"))
        writer.writerow(("Calmar Ratio", f"{self.result.calmar_computed:.4f}"))
        writer.writerow(("Max Drawdown", f"{self.result.max_drawdown_pct:.6f}"))
        writer.writerow(("Profit Factor", f"{self.result.profit_factor:.4f}"))
        writer.writerow(("Win Rate", f"{self.result.win_rate:.4f}"))
        return buf.getvalue()

    def to_pandas(self) -> Any:
        """Return the equity curve as a pandas DataFrame.

        Returns
        -------
        Any
            pandas DataFrame with columns ``index``, ``equity``,
            ``drawdown``, and optionally ``benchmark``.
        """
        import pandas as pd

        df = pd.DataFrame({
            "index": range(len(self.result.equity_curve)),
            "equity": self.result.equity_curve,
            "drawdown": self.result.drawdown_curve,
        })
        if self.result.benchmark_curve:
            df["benchmark"] = self.result.benchmark_curve
        return df

    def to_pnl_df(self) -> Any:
        """Return a DataFrame of individual trade P&L.

        Returns
        -------
        Any
            pandas DataFrame with trade-level data.
        """
        import pandas as pd

        records = []
        for t in self.result.trade_log:
            records.append({
                "trade_id": records.__len__() + 1,
                "instrument": t.instrument,
                "side": t.side.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "commission": t.commission,
                "holding_bars": t.holding_bars,
                "exit_reason": t.exit_reason.value,
            })
        return pd.DataFrame(records)
