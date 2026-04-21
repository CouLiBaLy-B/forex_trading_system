import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class EMA(TechnicalIndicator):
    """Exponential Moving Average - gives more weight to recent prices."""

    def __init__(self, period: int = 20):
        super().__init__("ema", {"period": period})
        self.period = period

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        close = np.array(prices.close())
        multiplier = 2 / (self.period + 1)
        values = [None] * (self.period - 1)
        ema = [close[0]]
        for i in range(1, len(close)):
            ema.append(ema[-1] * (1 - multiplier) + close[i] * multiplier)
        values.extend(ema[self.period - 1:])
        return IndicatorResult(name=self.name, values=values, timestamp=[])
