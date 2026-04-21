import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class StochasticOscillator(TechnicalIndicator):
    """Stochastic Oscillator - compares closing price to price range over time."""

    def __init__(self, k_period: int = 14, d_period: int = 3):
        super().__init__("stochastic", {"k_period": k_period, "d_period": d_period})
        self.k_period = k_period
        self.d_period = d_period

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        high = np.array(prices.high())
        low = np.array(prices.low())
        close = np.array(prices.close())
        k_vals = []
        for i in range(self.k_period - 1, len(close)):
            highest = np.max(high[i - self.k_period + 1:i + 1])
            lowest = np.min(low[i - self.k_period + 1:i + 1])
            k = ((close[i] - lowest) / (highest - lowest + 1e-10)) * 100
            k_vals.append(k)
        d_vals = np.convolve(k_vals, np.ones(self.d_period)/self.d_period, mode='valid')
        values = [None] * (self.k_period + self.d_period - 2) + list(zip(k_vals[self.d_period - 1:], d_vals))
        return IndicatorResult(name=self.name, values=values, timestamp=[])
