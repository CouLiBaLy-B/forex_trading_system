import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class RSI(TechnicalIndicator):
    """Relative Strength Index - momentum oscillator."""

    def __init__(self, period: int = 14):
        super().__init__("rsi", {"period": period})
        self.period = period

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        close = np.array(prices.close())
        delta = np.diff(close, prepend=close[0])
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        # Wilder RMA: first value is SMA, then exponential smoothing with alpha=1/period
        avg_gain = np.empty(len(close))
        avg_loss = np.empty(len(close))
        first_val = np.mean(gains[:self.period])
        avg_gain[:self.period] = first_val
        avg_loss[:self.period] = np.mean(losses[:self.period])

        alpha = 1.0 / self.period
        for i in range(self.period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (self.period - 1) + gains[i]) / self.period
            avg_loss[i] = (avg_loss[i-1] * (self.period - 1) + losses[i]) / self.period

        rs = np.where(avg_loss == 0, 100.0, np.where(avg_gain == 0, 0.0, avg_gain / avg_loss))
        rsi = 100.0 - (100.0 / rs)

        values = [None] * (self.period - 1) + rsi[self.period - 1:].tolist()
        return IndicatorResult(name=self.name, values=values, timestamp=[])
