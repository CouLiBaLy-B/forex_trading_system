import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class ATR(TechnicalIndicator):
    """Average True Range - measures market volatility."""

    def __init__(self, period: int = 14):
        super().__init__("atr", {"period": period})
        self.period = period

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        high = np.array(prices.high())
        low = np.array(prices.low())
        close = np.array(prices.close())
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
        atr = np.convolve(tr, np.ones(self.period)/self.period, mode='valid')
        values = [None] * (self.period - 1) + atr.tolist()
        return IndicatorResult(name=self.name, values=values, timestamp=[])
