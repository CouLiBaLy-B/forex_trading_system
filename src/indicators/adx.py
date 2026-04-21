import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class ADX(TechnicalIndicator):
    """Average Directional Index - measures trend strength."""

    def __init__(self, period: int = 14):
        super().__init__("adx", {"period": period})
        self.period = period

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        high = np.array(prices.high())
        low = np.array(prices.low())
        close = np.array(prices.close())
        dm_up = np.diff(high)
        dm_dn = -np.diff(low)
        dm_up = np.where((dm_up > dm_dn) & (dm_up > 0), dm_up, 0)
        dm_dn = np.where((dm_dn > dm_up) & (dm_dn > 0), dm_dn, 0)
        atr = np.convolve(np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:])),
                          np.ones(self.period)/self.period, mode='valid')
        di_up = np.convolve(dm_up, np.ones(self.period)/self.period, mode='valid') / (atr + 1e-10) * 100
        di_dn = np.convolve(dm_dn, np.ones(self.period)/self.period, mode='valid') / (atr + 1e-10) * 100
        dx = np.abs(di_up - di_dn) / (np.abs(di_up + di_dn) + 1e-10) * 100
        adx = np.convolve(dx, np.ones(self.period)/self.period, mode='valid')
        values = [None] * (2 * self.period - 2) + adx.tolist()
        return IndicatorResult(name=self.name, values=values, timestamp=[])
