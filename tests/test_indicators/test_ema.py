"""Tests for Exponential Moving Average (EMA)."""

import math
import pytest
from src.indicators.ema import EMA
from src.indicators.base import PriceSeries


class TestEMACompute:
    def test_basic(self, usd_jpy_prices: list[float]):
        ema = EMA(20)
        result = ema.compute(PriceSeries(usd_jpy_prices))
        values = result.values
        # First period-1 values should be None
        assert values[:19] == [None] * 19
        assert values[19] is not None
        # First EMA value = SMA of first period
        expected_sma = sum(usd_jpy_prices[:20]) / 20
        assert math.isclose(values[19], expected_sma, rel_tol=1e-9)

    def test_all_values_present_after_warmup(self, usd_jpy_prices: list[float]):
        ema = EMA(20)
        result = ema.compute(PriceSeries(usd_jpy_prices))
        assert len(result.values) == 200
        assert all(v is not None for v in result.values[20:])

    def test_short_series(self, short_prices: list[float]):
        ema = EMA(5)
        result = ema.compute(PriceSeries(short_prices))
        assert len(result.values) == 5
        assert result.values[4] is not None

    def test_period_larger_than_data(self, short_prices: list[float]):
        ema = EMA(10)
        result = ema.compute(PriceSeries(short_prices))
        assert all(v is None for v in result.values)

    def test_single_value(self, single_price: list[float]):
        ema = EMA(1)
        result = ema.compute(PriceSeries(single_price))
        assert result.values[0] == 100.0

    def test_constant_prices(self, constant_prices: list[float]):
        ema = EMA(10)
        result = ema.compute(PriceSeries(constant_prices))
        # EMA of constant = constant after warmup
        assert result.values[9] == 50.0
        assert all(v == 50.0 for v in result.values[9:])

    def test_empty_prices(self):
        ema = EMA(10)
        with pytest.raises(ValueError):
            ema.compute(PriceSeries([]))

    def test_result_properties(self, usd_jpy_prices: list[float]):
        ema = EMA(20)
        result = ema.compute(PriceSeries(usd_jpy_prices))
        assert result.name == "EMA20"
        assert result.period == 20

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            EMA(0).compute(PriceSeries([1.0, 2.0]))

    def test_negative_period(self):
        with pytest.raises(ValueError):
            EMA(-5).compute(PriceSeries([1.0, 2.0]))

    def test_windowed_ema(self, usd_jpy_prices: list[float]):
        ema = EMA(20)
        result = ema.compute(PriceSeries(usd_jpy_prices, windowed=True))
        assert result.windowed

    def test_ema_responds_to_trend(self, usd_jpy_prices: list[float]):
        """EMA should track price direction (less lag than SMA)."""
        ema = EMA(20)
        result = ema.compute(PriceSeries(usd_jpy_prices))
        values = result.values
        # Last EMA should be close to last price (smaller lag)
        # For 200 points with 20 period, EMA should be fairly responsive
        assert values[-1] is not None
        # EMA should be within reasonable range of current price
        assert abs(values[-1] - usd_jpy_prices[-1]) < 10.0


class EMAComparison:
    """Cross-comparison tests."""
    def test_ema_vs_sma_short_term(self, usd_jpy_prices: list[float]):
        """EMA should be closer to latest price than SMA (less lag)."""
        import numpy as np
        from src.indicators.sma import SMA

        ema = EMA(20)
        sma = SMA(20)
        ema_result = ema.compute(PriceSeries(usd_jpy_prices))
        sma_result = sma.compute(PriceSeries(usd_jpy_prices))

        ema_err = abs(ema_result.values[-1] - usd_jpy_prices[-1])
        sma_err = abs(sma_result.values[-1] - usd_jpy_prices[-1])
        assert ema_err < sma_err


class TestEMADocsExample:
    """Reproduce the docstring example."""
    def test_example_from_docs(self):
        ema = EMA(10)
        result = ema.compute(PriceSeries([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
        assert result.values[9] is not None
