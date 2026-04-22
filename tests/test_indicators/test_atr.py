"""Tests for Average True Range (ATR)."""

import math
import pytest
from src.indicators.atr import ATR
from src.indicators.base import OHLCData, PriceSeries


class TestATRCompute:
    def test_basic(self):
        ohlc_data = [
            OHLCData(100, 105, 98, 103),  # TR = 105-98 = 7
            OHLCData(103, 107, 101, 106),  # TR = 107-101 = 6
            OHLCData(106, 110, 104, 108),  # TR = 110-104 = 6
            OHLCData(108, 112, 105, 109),  # TR = 112-105 = 7
            OHLCData(109, 113, 107, 111),  # TR = 113-107 = 6
        ]
        atr = ATR(3)
        result = atr.compute(ohlc_data)
        assert len(result.values) == 5
        # First ATR = first TR
        assert result.values[0] == 7.0
        assert result.values[1] is not None
        assert result.values[2] is not None

    def test_atr_values_positive(self, usd_jpy_prices: list[float]):
        """ATR should always be positive for realistic data."""
        ohlc_data = [
            OHLCData(p * 1.001, p * 0.999, p * 0.995, p)
            for p in usd_jpy_prices
        ]
        atr = ATR(14)
        result = atr.compute(ohlc_data)
        for i, v in enumerate(result.values):
            if v is not None:
                assert v > 0, f"ATR value {v} at index {i} not positive"

    def test_all_values_after_warmup(self, usd_jpy_prices: list[float]):
        ohlc_data = [
            OHLCData(p * 1.001, p * 0.999, p * 0.995, p)
            for p in usd_jpy_prices
        ]
        atr = ATR(14)
        result = atr.compute(ohlc_data)
        assert len(result.values) == 200
        assert all(v is not None for v in result.values[14:])

    def test_constant_prices(self):
        """ATR of constant prices = 0."""
        ohlc_data = [OHLCData(100, 100, 100, 100)] * 30
        atr = ATR(14)
        result = atr.compute(ohlc_data)
        assert result.values[0] == 0.0
        assert all(v == 0.0 for v in result.values[14:])

    def test_single_bar(self):
        ohlc_data = [OHLCData(100, 105, 98, 103)]
        atr = ATR(14)
        result = atr.compute(ohlc_data)
        assert result.values[0] == 7.0  # first TR = high - low

    def test_empty_data(self):
        atr = ATR(14)
        with pytest.raises(ValueError):
            atr.compute(OHLCData([], []))

    def test_result_properties(self, usd_jpy_prices: list[float]):
        ohlc_data = [
            OHLCData(p * 1.001, p * 0.999, p * 0.995, p)
            for p in usd_jpy_prices
        ]
        atr = ATR(14)
        result = atr.compute(ohlc_data)
        assert result.name == "ATR14"
        assert result.period == 14

    def test_invalid_period(self):
        ohlc_data = [OHLCData(100, 105, 98, 103)]
        with pytest.raises(ValueError):
            ATR(0).compute(ohlc_data)

    def test_negative_period(self):
        ohlc_data = [OHLCData(100, 105, 98, 103)]
        with pytest.raises(ValueError):
            ATR(-5).compute(ohlc_data)

    def test_windowed_atr(self, usd_jpy_prices: list[float]):
        ohlc_data = [
            OHLCData(p * 1.001, p * 0.999, p * 0.995, p)
            for p in usd_jpy_prices
        ]
        atr = ATR(14)
        result = atr.compute(ohlc_data, windowed=True)
        assert result.windowed

    def test_first_tr_correct(self):
        """First TR should be high[0] - low[0], not using np.roll."""
        ohlc_data = [
            OHLCData(100, 108, 92, 100),  # TR = 108-92 = 16
            OHLCData(100, 105, 95, 100),  # TR = |105-95| = ... = 10
        ]
        atr = ATR(2)
        result = atr.compute(ohlc_data)
        assert result.values[0] == 16.0


class TestATRComparison:
    def test_wider_range_larger_atr(self):
        """Larger price swings => larger ATR."""
        ohlc_tight = [OHLCData(100, 100.1, 99.9, 100.05)] * 30
        ohlc_wide = [OHLCData(100, 110, 90, 100.05)] * 30
        atr_tight = ATR(14).compute(OHLCData(
            [o.high for o in ohlc_tight], [o.low for o in ohlc_tight]
        ))
        atr_wide = ATR(14).compute(OHLCData(
            [o.high for o in ohlc_wide], [o.low for o in ohlc_wide]
        ))
        assert atr_wide.values[14] > atr_tight.values[14]
