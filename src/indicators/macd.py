import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class MACD(TechnicalIndicator):
    """Moving Average Convergence Divergence - trend-following momentum indicator."""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("macd", {"fast": fast_period, "slow": slow_period, "signal": signal_period})
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        close = np.array(prices.close())
        ema_fast = [close[0]]
        ema_slow = [close[0]]
        mult_fast = 2 / (self.fast_period + 1)
        mult_slow = 2 / (self.slow_period + 1)
        for i in range(1, len(close)):
            ema_fast.append(ema_fast[-1] * (1 - mult_fast) + close[i] * mult_fast)
            ema_slow.append(ema_slow[-1] * (1 - mult_slow) + close[i] * mult_slow)
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_vals = np.convolve(macd_line, np.ones(self.signal_period)/self.signal_period, mode='valid')
        histogram = macd_line[self.signal_period:] - signal_vals
        values = [None] * (self.slow_period + self.signal_period) + list(zip(macd_line[self.slow_period:], signal_vals, histogram))
        return IndicatorResult(name=self.name, values=values, timestamp=[])
