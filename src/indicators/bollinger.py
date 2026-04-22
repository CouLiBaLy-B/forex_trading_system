import numpy as np
from .base import TechnicalIndicator, IndicatorResult, PriceSeries
from .ema import EMA
from .sma import SMA


def bollinger_bands(prices: list[float], period: int = 20, num_stddev: float = 2.0) -> dict[str, list[float | None]]:
    """Classic Bollinger Bands: SMA +- (num_stddev * rolling stddev).

    Args:
        prices: Close prices series.
        period: SMA period.
        num_stddev: Standard deviation multiplier (width factor).

    Returns:
        Dictionary with 'upper', 'middle', 'lower' bands.
    """
    sma = SMA(period)
    sma_result = sma.compute(PriceSeries(prices))

    upper: list[float | None] = []
    lower: list[float | None] = []
    middle: list[float | None] = []

    for i, mid in enumerate(sma_result.values):
        if mid is None:
            upper.append(None)
            lower.append(None)
            middle.append(None)
        else:
            window = prices[max(0, i - period + 1):i + 1]
            stddev = np.std(window, ddof=1)
            upper.append(mid + num_stddev * stddev)
            lower.append(mid - num_stddev * stddev)
            middle.append(mid)

    return {"upper": upper, "middle": middle, "lower": lower}


def bollinger_pct_b(prices: list[float], period: int = 20, num_stddev: float = 2.0) -> list[float]:
    """Bollinger %b indicator.

    Computes (price - lower_band) / (upper_band - lower_band).
    Values > 1.0 indicate price is above upper band; < 0.0 below lower band.
    """
    bands = bollinger_bands(prices, period, num_stddev)
    pct_b: list[float] = []

    for i in range(len(prices)):
        upper = bands["upper"][i]
        lower = bands["lower"][i]
        if upper is None or lower is None or upper == lower:
            pct_b.append(0.5)
        else:
            pct_b.append((prices[i] - lower) / (upper - lower))

    return pct_b
