"""Shared fixtures for forex_trading_system tests."""

import pytest


@pytest.fixture
def usd_jpy_prices() -> list[float]:
    """Typical USD/JPY price series (200 data points)."""
    import random
    random.seed(42)
    prices = [145.0 + sum(random.uniform(-0.15, 0.15) for _ in range(i)) for i in range(200)]
    return prices


@pytest.fixture
def eur_usd_prices() -> list[float]:
    """Typical EUR/USD price series (200 data points)."""
    import random
    random.seed(43)
    prices = [1.08 + sum(random.uniform(-0.001, 0.001) for _ in range(i)) for i in range(200)]
    return prices


@pytest.fixture
def short_prices() -> list[float]:
    """Short price series (under period length)."""
    return [10.0, 11.0, 12.0, 13.0, 14.0]


@pytest.fixture
def constant_prices() -> list[float]:
    """Constant price series (edge case: zero volatility)."""
    return [50.0] * 30


@pytest.fixture
def single_price() -> list[float]:
    """Single data point."""
    return [100.0]
