import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries

class VWAP(TechnicalIndicator):
    """Volume Weighted Average Price."""

    def __init__(self):
        super().__init__("vwap", {})

    def compute(self, prices: PriceSeries) -> IndicatorResult:
        typical = (np.array(prices.high()) + np.array(prices.low()) + np.array(prices.close())) / 3
        volume = np.array(prices.volume())
        cum_tp_vol = np.cumsum(typical * volume)
        cum_vol = np.cumsum(volume)
        values = (cum_tp_vol / (cum_vol + 1e-10)).tolist()
        return IndicatorResult(name=self.name, values=values, timestamp=[])
