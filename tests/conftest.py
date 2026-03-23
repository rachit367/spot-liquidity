"""
Shared pytest fixtures for the ICT bot test suite.
"""

import numpy as np
import pandas as pd
import pytest
from execution import TradeSignal


@pytest.fixture
def sample_ohlcv():
    """Generate a simple OHLCV DataFrame with 100 candles."""
    np.random.seed(42)
    n = 100
    base = 50000.0
    prices = base + np.cumsum(np.random.randn(n) * 100)

    data = {
        "open":  prices,
        "high":  prices + np.abs(np.random.randn(n) * 50),
        "low":   prices - np.abs(np.random.randn(n) * 50),
        "close": prices + np.random.randn(n) * 30,
    }
    df = pd.DataFrame(data)
    # Ensure high >= open,close and low <= open,close
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 10
    df["low"]  = df[["open", "low", "close"]].min(axis=1) - 10
    return df


@pytest.fixture
def bullish_ohlcv():
    """Generate a clearly bullish OHLCV DataFrame."""
    np.random.seed(7)
    n = 100
    base = 50000.0
    # Uptrend: each candle higher than the last
    trend = np.linspace(0, 3000, n)
    noise = np.random.randn(n) * 30

    opens  = base + trend + noise
    closes = opens + np.abs(np.random.randn(n) * 20) + 5  # bullish candles
    highs  = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 15) + 5
    lows   = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 15) - 5

    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


@pytest.fixture
def bearish_ohlcv():
    """Generate a clearly bearish OHLCV DataFrame."""
    np.random.seed(12)
    n = 100
    base = 50000.0
    trend = np.linspace(0, -3000, n)
    noise = np.random.randn(n) * 30

    opens  = base + trend + noise
    closes = opens - np.abs(np.random.randn(n) * 20) - 5  # bearish candles
    highs  = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 15) + 5
    lows   = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 15) - 5

    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


@pytest.fixture
def sample_signal():
    """Create a sample bullish trade signal."""
    return TradeSignal(
        symbol="BTCUSD",
        entry=50000.0,
        stop_loss=49500.0,
        take_profit=51000.0,
        direction="long",
        reason="Test signal",
    )


@pytest.fixture
def sample_bear_signal():
    """Create a sample bearish trade signal."""
    return TradeSignal(
        symbol="BTCUSD",
        entry=50000.0,
        stop_loss=50500.0,
        take_profit=49000.0,
        direction="short",
        reason="Test bear signal",
    )


@pytest.fixture
def tmp_trades_csv(tmp_path):
    """Return a temporary path for test trades CSV."""
    return tmp_path / "trades_ml.csv"
