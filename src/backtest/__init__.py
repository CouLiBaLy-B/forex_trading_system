"""Backtesting engine for forex trading strategies.

This package provides an event-driven backtesting framework with realistic
market microstructure simulation (spread, slippage, commission), comprehensive
performance metrics, and reporting utilities.

Modules
-------
models :
    Pydantic DTOs: BacktestConfig, BacktestResult, BacktestTrade.
engine :
    BacktestEngine: the core event-driven backtesting loop.
report :
    BacktestReporter: CSV / JSON / equity curve / benchmark outputs.

Quick start
-----------
>>> from src.backtest import BacktestEngine, BacktestResult, BacktestConfig
>>> config = BacktestConfig(strategy_name="ma_cross", instrument="EUR/USD")
>>> engine = BacktestEngine(config, my_strategy, ohlcv_data)
>>> result: BacktestResult = engine.run()
>>> print(f"Sharpe: {result.sharpe_ratio:.4f}")
"""

from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestConfig, BacktestFoldResult, BacktestResult, BacktestTrade
from src.backtest.report import BacktestReporter

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestFoldResult",
    "BacktestResult",
    "BacktestReporter",
    "BacktestTrade",
]
