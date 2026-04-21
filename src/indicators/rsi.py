import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class RSI(TechnicalIndicator):
    """Relative Strength Index - momentum oscillator measuring speed of price changes."""

    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__("rsi", {"period": period, "overbought": overbought, "oversold": oversold})
        self.period = period

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        close = np.array(prices.close())
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(self.period)/self.period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(self.period)/self.period, mode='valid')
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_vals = 100 - (100 / (1 + rs))
        values = [None] * self.period + rsi_vals.tolist()
        return IndicatorResult(name=self.name, values=values, timestamp=[])
