"""Tests for Simple Moving Average (SMA)."""

import math
import pytest
from src.indicators.sma import SMA
from src.indicators.base import PriceSeries


class TestSMACompute:
    def test_basic(self, usd_jpy_prices: list[float]):
        sma = SMA(20)
        result = sma.compute(PriceSeries(usd_jpy_prices))
        values = result.values
        # First period-1 values should be None
        assert values[:19] == [None] * 19
        assert values[19] is not None
        # Value at index 19 = SMA of first 20 prices
        expected = sum(usd_jpy_prices[:20]) / 20
        assert math.isclose(values[19], expected, rel_tol=1e-9)

    def test_all_values_present_after_warmup(self, usd_jpy_prices: list[float]):
        sma = SMA(20)
        result = sma.compute(PriceSeries(usd_jpy_prices))
        assert len(result.values) == 200
        assert result.values[:19] == [None] * 19
        assert all(v is not None for v in result.values[20:])

    def test_windowed_sma(self, usd_jpy_prices: list[float]):
        sma = SMA(20)
        result = sma.compute(PriceSeries(usd_jpy_prices, windowed=True))
        assert result.windowed
        assert len(result.values) == 200

    def test_short_series(self, short_prices: list[float]):
        sma = SMA(5)
        result = sma.compute(PriceSeries(short_prices))
        assert len(result.values) == 5
        assert result.values[4] == 12.0  # mean of [10,11,12,13,14]

    def test_period_larger_than_data(self, short_prices: list[float]):
        sma = SMA(10)
        result = sma.compute(PriceSeries(short_prices))
        assert all(v is None for v in result.values)

    def test_single_value(self, single_price: list[float]):
        sma = SMA(1)
        result = sma.compute(PriceSeries(single_price))
        assert result.values[0] == 100.0

    def test_constant_prices(self, constant_prices: list[float]):
        sma = SMA(10)
        result = sma.compute(PriceSeries(constant_prices))
        assert all(v == 50.0 for v in result.values[9:])

    def test_result_properties(self, usd_jpy_prices: list[float]):
        sma = SMA(20)
        result = sma.compute(PriceSeries(usd_jpy_prices))
        assert result.name == "SMA20"
        assert result.period == 20
        assert len(result) == 200

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            SMA(0).compute(PriceSeries([1.0, 2.0]))

    def test_negative_period(self):
        with pytest.raises(ValueError):
            SMA(-5).compute(PriceSeries([1.0, 2.0]))

    def test_empty_prices(self):
        sma = SMA(10)
        with pytest.raises(ValueError):
            sma.compute(PriceSeries([]))


class TestSMADocsExample:
    """Reproduce the docstring calculation."""
    def test_example_from_docs(self):
        sma = SMA(10)
        result = sma.compute(PriceSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
        assert result.values[9] == 5.5
