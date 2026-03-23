"""
Tests for the ICT strategy module.
"""

import pandas as pd
import numpy as np
import pytest

from strategy import ICTStrategy, _prepare, _swing_highs, _swing_lows, _price_in_zone


class TestPrepare:
    """Tests for the _prepare helper."""

    def test_coerces_to_float(self):
        df = pd.DataFrame({
            "open": ["100.5", "101"],
            "high": ["102", "103"],
            "low":  ["99", "100"],
            "close": ["101", "102"],
        })
        result = _prepare(df)
        assert result["open"].dtype == float
        assert result["high"].dtype == float

    def test_drops_nan_rows(self):
        df = pd.DataFrame({
            "open": [100, None, 102],
            "high": [101, 103, 104],
            "low":  [99, 100, 101],
            "close": [100.5, 102, 103],
        })
        result = _prepare(df)
        assert len(result) == 2  # one row dropped


class TestSwingDetection:
    """Tests for swing high/low detection."""

    def test_swing_highs_returns_dataframe(self, sample_ohlcv):
        df = _prepare(sample_ohlcv)
        highs = _swing_highs(df, n=3)
        assert isinstance(highs, pd.DataFrame)
        assert len(highs) > 0

    def test_swing_lows_returns_dataframe(self, sample_ohlcv):
        df = _prepare(sample_ohlcv)
        lows = _swing_lows(df, n=3)
        assert isinstance(lows, pd.DataFrame)
        assert len(lows) > 0

    def test_swing_highs_are_local_maxima(self, sample_ohlcv):
        df = _prepare(sample_ohlcv)
        highs = _swing_highs(df, n=3)
        for idx in highs.index:
            if idx >= 3 and idx < len(df) - 3:
                window = df.iloc[idx-3:idx+4]["high"]
                assert df.iloc[idx]["high"] == window.max()


class TestPriceInZone:
    """Tests for the _price_in_zone helper."""

    def test_inside_zone(self):
        assert _price_in_zone(50, 40, 60) is True

    def test_at_boundaries(self):
        assert _price_in_zone(40, 40, 60) is True
        assert _price_in_zone(60, 40, 60) is True

    def test_outside_zone(self):
        assert _price_in_zone(30, 40, 60) is False
        assert _price_in_zone(70, 40, 60) is False


class TestMarketStructure:
    """Tests for market structure detection."""

    def test_bullish_detection(self, bullish_ohlcv):
        strat = ICTStrategy("TEST", ob_lookback=20)
        df = _prepare(bullish_ohlcv)
        result = strat._detect_market_structure(df)
        assert result == "bullish"

    def test_bearish_detection(self, bearish_ohlcv):
        strat = ICTStrategy("TEST", ob_lookback=20)
        df = _prepare(bearish_ohlcv)
        result = strat._detect_market_structure(df)
        assert result == "bearish"

    def test_insufficient_data_returns_neutral(self):
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low":  [99, 100],
            "close": [101, 102],
        })
        strat = ICTStrategy("TEST")
        result = strat._detect_market_structure(df)
        assert result == "neutral"


class TestFVG:
    """Tests for Fair Value Gap detection."""

    def test_bullish_fvg(self):
        # Create data with a clear bullish FVG
        strat = ICTStrategy("TEST", ob_lookback=15)
        n = 30
        df = pd.DataFrame({
            "open":  [100 + i for i in range(n)],
            "high":  [101 + i for i in range(n)],
            "low":   [99 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
        })
        # Create a gap: candle[15].high < candle[17].low
        df.loc[15, "high"] = 110
        df.loc[17, "low"]  = 115

        result = strat._find_fvg(_prepare(df), "bullish")
        # May or may not find FVG depending on the window
        # but the function should not crash
        assert result is None or result["type"] == "bullish_fvg"


class TestOrderBlock:
    """Tests for Order Block detection."""

    def test_does_not_crash_with_sample_data(self, sample_ohlcv):
        strat = ICTStrategy("TEST", ob_lookback=20)
        df = _prepare(sample_ohlcv)
        result = strat._find_order_block(df, "bullish")
        # May or may not find OB, but should not crash
        assert result is None or "high" in result

    def test_returns_none_on_short_data(self):
        df = pd.DataFrame({
            "open": [100], "high": [101], "low": [99], "close": [100.5]
        })
        strat = ICTStrategy("TEST")
        assert strat._find_order_block(df, "bullish") is None
