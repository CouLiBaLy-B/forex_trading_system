import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class SMA(TechnicalIndicator):
    """Simple Moving Average - averages prices over a rolling window."""

    def __init__(self, period: int = 20):
        super().__init__("sma", {"period": period})
        self.period = period

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        close = np.array(prices.close())
        values = [None] * (self.period - 1) + np.convolve(close, np.ones(self.period)/self.period, mode='valid').tolist()
        return IndicatorResult(name=self.name, values=values, timestamp=[])
