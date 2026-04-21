import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class BollingerBands(TechnicalIndicator):
    """Bollinger Bands - volatility bands around a moving average."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("bollinger", {"period": period, "std_dev": std_dev})
        self.period = period
        self.std_dev = std_dev

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        close = np.array(prices.close())
        middle = np.convolve(close, np.ones(self.period)/self.period, mode='valid')
        std = np.empty_like(middle)
        for i in range(len(middle)):
            std[i] = np.std(close[i:self.period + i])
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        values = [None] * (self.period - 1) + [
            (u, m, l) for u, m, l in zip(upper, middle, lower)
        ]
        return IndicatorResult(name=self.name, values=values, timestamp=[])
