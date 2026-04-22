"""Tests for Bollinger Bands."""

import math
import pytest
from src.indicators.bollinger import bollinger_bands, bollinger_pct_b


class TestBollingerBands:
    def test_basic(self, usd_jpy_prices: list[float]):
        result = bollinger_bands(usd_jpy_prices, period=20, num_stddev=2.0)
        assert len(result["upper"]) == 200
        assert len(result["middle"]) == 200
        assert len(result["lower"]) == 200
        # Warmup values
        assert result["upper"][:19] == [None] * 19
        assert result["middle"][:19] == [None] * 19
        assert result["lower"][:19] == [None] * 19
        # After warmup, upper > middle > lower
        for i in range(20, 200):
            assert result["upper"][i] is not None
            assert result["middle"][i] is not None
            assert result["lower"][i] is not None
            assert result["upper"][i] > result["middle"][i] > result["lower"][i]

    def test_middle_equals_sma(self, usd_jpy_prices: list[float]):
        """Middle band = SMA for Bollinger Bands."""
        from src.indicators.sma import SMA

        period = 20
        result = bollinger_bands(usd_jpy_prices, period=period)
        sma = SMA(period).compute(PriceSeries(usd_jpy_prices))

        for i in range(period, 200):
            assert math.isclose(result["middle"][i], sma.values[i], rel_tol=1e-9)

    def test_width_with_stddev(self, usd_jpy_prices: list[float]):
        """Upper band = middle + num_stddev * rolling_stddev."""
        import numpy as np
        from src.indicators.sma import SMA

        period = 20
        num_stddev = 2.0
        result = bollinger_bands(usd_jpy_prices, period=period, num_stddev=num_stddev)

        sma = SMA(period).compute(PriceSeries(usd_jpy_prices))
        for i in range(period, min(200, period + 5)):
            window = usd_jpy_prices[max(0, i - period + 1):i + 1]
            expected_upper = sma.values[i] + num_stddev * np.std(window, ddof=1)
            assert math.isclose(result["upper"][i], expected_upper, rel_tol=1e-6), f"Mismatch at index {i}"

    def test_custom_stddev_multiplier(self, usd_jpy_prices: list[float]):
        """Different num_stddev produces different widths."""
        bands_1 = bollinger_bands(usd_jpy_prices, period=20, num_stddev=1.0)
        bands_2 = bollinger_bands(usd_jpy_prices, period=20, num_stddev=3.0)

        # Higher stddev => wider bands
        for i in range(20, 200):
            assert bands_2["upper"][i] > bands_1["upper"][i]
            assert bands_2["lower"][i] < bands_1["lower"][i]

    def test_constant_prices(self, constant_prices: list[float]):
        """Bollinger Bands of constant prices = flat middle, width = 0."""
        result = bollinger_bands(constant_prices, period=10)
        assert all(v == 50.0 for v in result["middle"][9:])
        # Stddev = 0 => upper = lower = middle
        assert all(math.isclose(v, 50.0, abs_tol=1e-9) for v in result["upper"][10:])
        assert all(math.isclose(v, 50.0, abs_tol=1e-9) for v in result["lower"][10:])

    def test_short_series(self, short_prices: list[float]):
        result = bollinger_bands(short_prices, period=5)
        assert len(result["upper"]) == 5
        assert result["upper"][4] is not None

    def test_period_larger_than_data(self, short_prices: list[float]):
        result = bollinger_bands(short_prices, period=10)
        assert all(v is None for v in result["upper"])
        assert all(v is None for v in result["lower"])

    def test_single_value(self, single_price: list[float]):
        result = bollinger_bands(single_price, period=1)
        assert result["upper"][0] == 100.0
        assert result["lower"][0] == 100.0

    def test_empty_prices(self):
        with pytest.raises(ValueError):
            bollinger_bands([], period=20)

    def test_width_positive(self, usd_jpy_prices: list[float]):
        """Width = upper - lower should be non-negative."""
        result = bollinger_bands(usd_jpy_prices, period=20)
        for i in range(20, 200):
            width = result["upper"][i] - result["lower"][i]
            assert width >= 0, f"Negative width {width} at index {i}"


class TestBollingerPctB:
    def test_basic(self, usd_jpy_prices: list[float]):
        result = bollinger_pct_b(usd_jpy_prices, period=20)
        assert len(result) == 200

    def test_pct_b_for_price_near_middle(self, usd_jpy_prices: list[float]):
        """Price near middle band => pct_b ~ 0.5."""
        # Create prices that stay near the SMA
        from src.indicators.sma import SMA
        sma_result = SMA(20).compute(PriceSeries(usd_jpy_prices))
        near_middle = [sma_result.values[i] if sma_result.values[i] else usd_jpy_prices[i]
                       for i in range(len(usd_jpy_prices))]

        pct_b = bollinger_pct_b(near_middle, period=20)
        # Values near middle should be close to 0.5
        for i in range(20, min(200, 20 + 10)):
            if pct_b[i] is not None:
                assert -0.1 < pct_b[i] < 0.9

    def test_pct_b_above_upper(self):
        """Price well above upper band => pct_b > 1."""
        prices = [50.0 + i * 2.0 for i in range(50)]  # strong uptrend
        pct_b = bollinger_pct_b(prices, period=20)
        # Last value should be above 1
        assert pct_b[-1] > 1.0

    def test_pct_b_below_lower(self):
        """Price well below lower band => pct_b < 0."""
        prices = [100.0 - i * 2.0 for i in range(50)]  # strong downtrend
        pct_b = bollinger_pct_b(prices, period=20)
        assert pct_b[-1] < 0.0

    def test_pct_b_in_range(self, usd_jpy_prices: list[float]):
        """Most normal prices => 0 < pct_b < 1."""
        pct_b = bollinger_pct_b(usd_jpy_prices, period=20)
        in_range = [v for v in pct_b[20:] if 0 < v < 1]
        # At least some should be in the band
        assert len(in_range) > 0

    def test_pct_b_constant(self, constant_prices: list[float]):
        """Constant prices => pct_b = 0.5 (width = 0, default to middle)."""
        pct_b = bollinger_pct_b(constant_prices, period=10)
        assert all(v == 0.5 for v in pct_b[10:])

    def test_pct_b_short_series(self, short_prices: list[float]):
        pct_b = bollinger_pct_b(short_prices, period=5)
        assert len(pct_b) == 5
        assert pct_b[-1] is not None

    def test_pct_b_empty(self):
        with pytest.raises(ValueError):
            bollinger_pct_b([], period=20)
