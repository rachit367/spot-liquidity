"""
Tests for the ML feature engineering module.
"""

import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import extract, FEATURE_NAMES, N_FEATURES


class TestExtract:
    """Tests for the main extract() function."""

    def test_returns_correct_number_of_features(self, sample_ohlcv, sample_signal):
        vec = extract(sample_ohlcv, sample_signal)
        assert isinstance(vec, np.ndarray)
        assert len(vec) == N_FEATURES

    def test_features_are_finite(self, sample_ohlcv, sample_signal):
        vec = extract(sample_ohlcv, sample_signal)
        assert np.all(np.isfinite(vec))

    def test_feature_names_count_matches(self):
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_bearish_signal_produces_features(self, sample_ohlcv, sample_bear_signal):
        vec = extract(sample_ohlcv, sample_bear_signal)
        assert isinstance(vec, np.ndarray)
        assert len(vec) == N_FEATURES

    def test_short_data_still_works(self, sample_signal):
        """Feature extraction should handle minimal data gracefully."""
        df = pd.DataFrame({
            "open":  np.random.randn(25) * 100 + 50000,
            "high":  np.random.randn(25) * 100 + 50100,
            "low":   np.random.randn(25) * 100 + 49900,
            "close": np.random.randn(25) * 100 + 50000,
        })
        df["high"] = df[["open", "high", "close"]].max(axis=1) + 5
        df["low"]  = df[["open", "low", "close"]].min(axis=1) - 5

        vec = extract(df, sample_signal)
        assert len(vec) == N_FEATURES
        assert np.all(np.isfinite(vec))

    def test_consistent_output(self, sample_ohlcv, sample_signal):
        """Two calls with the same data should produce the same features."""
        vec1 = extract(sample_ohlcv.copy(), sample_signal)
        vec2 = extract(sample_ohlcv.copy(), sample_signal)
        np.testing.assert_array_almost_equal(vec1, vec2)
