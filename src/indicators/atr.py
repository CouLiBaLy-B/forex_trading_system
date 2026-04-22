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

        # First TR = high - low (no prior close)
        # Subsequent TRs use diff to avoid np.roll wrap-around bug
        tr0 = high[0] - low[0]
        tr_rest = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        )
        tr = np.concatenate(([tr0], tr_rest))

        # Wilder smoothing for ATR
        atr = np.empty(len(tr))
        atr[:self.period] = np.mean(tr[:self.period])
        for i in range(self.period, len(tr)):
            atr[i] = (atr[i-1] * (self.period - 1) + tr[i]) / self.period

        values = [None] * (self.period - 1) + atr[self.period - 1:].tolist()
        return IndicatorResult(name=self.name, values=values, timestamp=[])
