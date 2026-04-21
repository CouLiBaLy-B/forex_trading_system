import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class DonchianChannel(TechnicalIndicator):
    """Donchian Channel - highest high and lowest low over a period."""

    def __init__(self, period: int = 20):
        super().__init__("donchian", {"period": period})
        self.period = period

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        high = np.array(prices.high())
        low = np.array(prices.low())
        upper = [np.max(high[i - self.period + 1:i + 1]) for i in range(self.period - 1, len(high))]
        lower = [np.min(low[i - self.period + 1:i + 1]) for i in range(self.period - 1, len(low))]
        middle = [(u + l) / 2 for u, l in zip(upper, lower)]
        values = [None] * (self.period - 1) + list(zip(upper, middle, lower))
        return IndicatorResult(name=self.name, values=values, timestamp=[])
