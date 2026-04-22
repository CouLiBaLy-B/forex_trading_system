import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class MACD(TechnicalIndicator):
    """Moving Average Convergence Divergence."""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("macd", {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
        })
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def _ema(self, arr: np.ndarray, period: int) -> list[float]:
        multiplier = 2 / (period + 1)
        ema = [arr[0]]
        for i in range(1, len(arr)):
            ema.append(ema[-1] * (1 - multiplier) + arr[i] * multiplier)
        return ema

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        close = np.array(prices.close())
        fast_ema = self._ema(close, self.fast_period)
        slow_ema = self._ema(close, self.slow_period)
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        signal_line = self._ema(np.array(macd_line), self.signal_period)
        histogram = [m - s for m, s in zip(macd_line, signal_line)]

        pad = max(self.fast_period, self.slow_period, self.signal_period)
        values = [None] * (pad - 1) + [
            (macd_line[i], signal_line[i], histogram[i])
            for i in range(pad - 1, len(close))
        ]
        return IndicatorResult(name=self.name, values=values, timestamp=[])
