"""Tests for MACD indicator."""

import math
import pytest
from src.indicators.macd import MACD, compute_crossover
from src.indicators.base import PriceSeries


class TestMACDCompute:
    def test_basic(self, usd_jpy_prices: list[float]):
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.compute(PriceSeries(usd_jpy_prices))
        # Should have 3 sub-indicators
        assert len(result.values) == 3
        # First indicator (MACD line) should have warmup values
        macd_line = result.values[0]
        assert macd_line is not None
        assert len(macd_line) == 200

    def test_macd_line_has_values_after_warmup(self, usd_jpy_prices: list[float]):
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.compute(PriceSeries(usd_jpy_prices))
        macd_line = result.values[0]
        # After period slow=26, values should be present
        assert all(v is not None for v in macd_line[26:])

    def test_signal_line_has_values(self, usd_jpy_prices: list[float]):
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.compute(PriceSeries(usd_jpy_prices))
        signal_line = result.values[1]
        # Signal needs signal_period more warmup on top of slow period
        assert all(v is not None for v in signal_line[26 + 9 - 1:])

    def test_histogram_has_values(self, usd_jpy_prices: list[float]):
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.compute(PriceSeries(usd_jpy_prices))
        histogram = result.values[2]
        assert all(v is not None for v in histogram[26 + 9 - 1:])

    def test_histogram_equals_macd_minus_signal(self, usd_jpy_prices: list[float]):
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.compute(PriceSeries(usd_jpy_prices))
        macd_line = result.values[0]
        signal_line = result.values[1]
        histogram = result.values[2]
        for i in range(40, 200):
            expected = macd_line[i] - signal_line[i]
            assert math.isclose(histogram[i], expected, rel_tol=1e-9)

    def test_short_series(self, short_prices: list[float]):
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.compute(PriceSeries(short_prices))
        assert len(result.values) == 3
        assert all(len(v) == 5 for v in result.values)

    def test_period_larger_than_data(self, short_prices: list[float]):
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.compute(PriceSeries(short_prices))
        assert result.values[0] == [None] * 5

    def test_single_value(self, single_price: list[float]):
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.compute(PriceSeries(single_price))
        assert result.values[0] == [None]

    def test_empty_prices(self):
        macd = MACD(fast=12, slow=26, signal=9)
        with pytest.raises(ValueError):
            macd.compute(PriceSeries([]))

    def test_result_properties(self, usd_jpy_prices: list[float]):
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.compute(PriceSeries(usd_jpy_prices))
        assert result.name == "MACD12_26_9"
        assert len(result) == 3
        assert len(result) == 3  # MACD line, signal line, histogram

    def test_invalid_fast(self):
        with pytest.raises(ValueError):
            MACD(fast=0, slow=26, signal=9).compute(PriceSeries([1.0]))

    def test_fast_larger_than_slow(self):
        with pytest.raises(ValueError):
            MACD(fast=30, slow=12, signal=9).compute(PriceSeries([1.0]))

    def test_invalid_signal(self):
        with pytest.raises(ValueError):
            MACD(fast=12, slow=26, signal=0).compute(PriceSeries([1.0]))

    def test_windowed_macd(self, usd_jpy_prices: list[float]):
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.compute(PriceSeries(usd_jpy_prices, windowed=True))
        assert result.windowed

    def test_macd_values_consistent_with_components(self, usd_jpy_prices: list[float]):
        """Verify MACD line = EMA_fast - EMA_slow."""
        from src.indicators.ema import EMA

        macd_inst = MACD(fast=12, slow=26, signal=9)
        result = macd_inst.compute(PriceSeries(usd_jpy_prices))

        ema_fast = EMA(12).compute(PriceSeries(usd_jpy_prices))
        ema_slow = EMA(26).compute(PriceSeries(usd_jpy_prices))

        macd_line = result.values[0]
        for i in range(26, min(200, 26 + 5)):
            expected = ema_fast.values[i] - ema_slow.values[i]
            assert math.isclose(macd_line[i], expected, rel_tol=1e-6), f"Mismatch at index {i}"


class TestCrossOver:
    def test_bullish_crossover(self):
        """MACD crosses above signal => bullish."""
        macd_hist = [None, None, -0.5, -0.3, 0.1, 0.3]
        signal_hist = [None, None, -0.6, -0.4, -0.1, 0.2]
        result = compute_crossover(macd_hist, signal_hist)
        assert result == "bullish"

    def test_bearish_crossover(self):
        """MACD crosses below signal => bearish."""
        macd_hist = [None, None, 0.5, 0.3, 0.1, -0.1]
        signal_hist = [None, None, 0.4, 0.2, 0.0, -0.2]
        result = compute_crossover(macd_hist, signal_hist)
        assert result == "bearish"

    def test_no_crossover(self):
        """No crossover => None."""
        macd_hist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        signal_hist = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        result = compute_crossover(macd_hist, signal_hist)
        assert result is None

    def test_all_none(self):
        result = compute_crossover([None, None], [None, None])
        assert result is None

    def test_too_short(self):
        result = compute_crossover([0.1], [0.0])
        assert result is None

    def test_equal_values(self):
        """Equal values at crossover point => no signal."""
        macd_hist = [None, None, -0.1, 0.0, 0.1]
        signal_hist = [None, None, 0.0, 0.0, 0.1]
        result = compute_crossover(macd_hist, signal_hist)
        assert result is None
