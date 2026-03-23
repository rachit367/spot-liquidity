"""
Tests for the dataset builder module.
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from ml.dataset_builder import append_trade, load_trades, build_dataset, CSV_COLUMNS
from ml.feature_engineering import N_FEATURES


@pytest.fixture
def mock_csv(tmp_path):
    """Redirect TRADES_CSV to a temp file."""
    csv_path = tmp_path / "trades_ml.csv"
    with patch("ml.dataset_builder.TRADES_CSV", csv_path):
        yield csv_path


class TestAppendTrade:
    """Tests for append_trade."""

    def test_creates_file_on_first_trade(self, mock_csv):
        record = {
            "trade_id": "test_001",
            "timestamp": "2024-01-01T00:00:00Z",
            "symbol": "BTCUSD",
            "direction": "long",
            "entry_price": 50000,
            "exit_price": 51000,
            "result": 1,
            "features": np.random.randn(N_FEATURES),
        }
        append_trade(record)
        assert mock_csv.exists()

    def test_appends_multiple_trades(self, mock_csv):
        for i in range(5):
            record = {
                "trade_id": f"test_{i:03d}",
                "timestamp": f"2024-01-0{i+1}T00:00:00Z",
                "symbol": "BTCUSD",
                "direction": "long",
                "result": i % 2,
                "entry_price": 50000 + i * 100,
                "exit_price": 50100 + i * 100,
                "features": np.random.randn(N_FEATURES),
            }
            append_trade(record)

        df = pd.read_csv(mock_csv)
        assert len(df) == 5

    def test_features_serialized_correctly(self, mock_csv):
        features = np.array([1.0, 2.0, 3.0] + [0.0] * (N_FEATURES - 3))
        record = {
            "trade_id": "test_feat",
            "result": 1,
            "features": features,
        }
        append_trade(record)

        df = pd.read_csv(mock_csv)
        parsed = json.loads(df.iloc[0]["features_json"])
        assert len(parsed) == N_FEATURES
        assert abs(parsed[0] - 1.0) < 0.001


class TestLoadTrades:
    """Tests for load_trades."""

    def test_returns_none_when_no_file(self, mock_csv):
        result = load_trades(min_rows=1)
        assert result is None

    def test_loads_completed_trades(self, mock_csv):
        for i in range(15):
            append_trade({
                "trade_id": f"test_{i:03d}",
                "timestamp": f"2024-01-{i+1:02d}T00:00:00Z",
                "symbol": "BTCUSD",
                "direction": "long",
                "result": i % 2,
                "entry_price": 50000,
                "exit_price": 51000,
                "features": np.random.randn(N_FEATURES),
            })

        df = load_trades(min_rows=10)
        assert df is not None
        assert len(df) >= 10


class TestBuildDataset:
    """Tests for build_dataset."""

    def test_builds_from_trades(self, mock_csv):
        np.random.seed(42)
        for i in range(30):
            append_trade({
                "trade_id": f"test_{i:03d}",
                "timestamp": f"2024-01-{(i % 28) + 1:02d}T00:00:00Z",
                "symbol": "BTCUSD",
                "direction": "long" if i % 2 else "short",
                "result": i % 2,
                "entry_price": 50000 + i * 10,
                "exit_price": 50100 + i * 10,
                "features": np.random.randn(N_FEATURES),
            })

        df = load_trades(min_rows=10)
        ds = build_dataset(df, test_frac=0.2)

        assert "X_train" in ds
        assert "X_test" in ds
        assert "y_train" in ds
        assert "y_test" in ds
        assert ds["X_train"].shape[1] == N_FEATURES
        assert len(ds["y_train"]) + len(ds["y_test"]) <= 30
