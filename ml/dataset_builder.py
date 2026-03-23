"""
Dataset builder — converts stored trade records into ML-ready (X, y) arrays.

Trade storage
-------------
Primary file : logs/trades_ml.csv
Schema (columns):
  trade_id, timestamp, symbol, direction, entry_price, exit_price,
  sl, tp, quantity, pnl, pnl_pct, rr_achieved, result,
  strategy, duration_bars, model_confidence, model_prediction,
  features_json

``result`` is 1 for win, 0 for loss/breakeven.
``features_json`` is a JSON array of 26 floats (the feature snapshot at entry).

No data leakage guarantee
--------------------------
• Features are extracted and stored at entry time (see api/server.py).
• The dataset_builder only reads what was stored — it never re-computes
  features from the full price history after the fact.
• Train/test split is time-ordered (chronological), never shuffled.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.feature_engineering import FEATURE_NAMES, N_FEATURES

logger = logging.getLogger(__name__)

TRADES_CSV = Path(__file__).resolve().parent.parent / "logs" / "trades_ml.csv"

# CSV column order (append-only schema — do not reorder)
CSV_COLUMNS = [
    "trade_id",
    "timestamp",
    "symbol",
    "direction",
    "entry_price",
    "exit_price",
    "sl",
    "tp",
    "quantity",
    "pnl",
    "pnl_pct",
    "rr_achieved",
    "result",          # 1 = win, 0 = loss / breakeven
    "strategy",
    "duration_bars",
    "model_confidence",
    "model_prediction",
    "features_json",   # JSON array of N_FEATURES floats
]


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers
# ─────────────────────────────────────────────────────────────────────────────

def append_trade(record: dict) -> None:
    """
    Append one completed trade to both CSV and SQLite.

    Missing columns are filled with None/empty.
    Creates the file with a header row if it does not exist.
    """
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)

    row = {col: record.get(col) for col in CSV_COLUMNS}

    # Serialise features vector to JSON string
    if "features" in record and row["features_json"] is None:
        feat = record["features"]
        if hasattr(feat, "tolist"):
            feat = feat.tolist()
        row["features_json"] = json.dumps([round(float(v), 6) for v in feat])

    # Write to CSV (backwards compatible)
    df_row = pd.DataFrame([row], columns=CSV_COLUMNS)
    header = not TRADES_CSV.exists()
    df_row.to_csv(TRADES_CSV, mode="a", header=header, index=False)

    # Write to SQLite
    try:
        from ml.database import get_db, insert_trade as db_insert
        db_insert(get_db(), row)
    except Exception as exc:
        logger.debug("SQLite write failed (non-critical): %s", exc)

    logger.debug("Trade appended (result=%s)", row["result"])


# ─────────────────────────────────────────────────────────────────────────────
# Read helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_trades(min_rows: int = 10) -> Optional[pd.DataFrame]:
    """
    Load completed trades from SQLite (preferred) or CSV fallback.

    Returns None when fewer than ``min_rows`` complete trades exist.
    """
    # Try SQLite first
    try:
        from ml.database import ensure_db_ready, query_trades
        conn = ensure_db_ready()
        df = query_trades(conn, min_rows=min_rows)
        if df is not None:
            df = df[df["result"].isin([0, 1])].copy()
            df["result"] = df["result"].astype(int)
            if len(df) >= min_rows:
                return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        logger.debug("SQLite read failed, falling back to CSV: %s", exc)

    # Fallback to CSV
    if not TRADES_CSV.exists():
        logger.info("No trade data found")
        return None

    df = pd.read_csv(TRADES_CSV, parse_dates=["timestamp"])
    df = df[df["result"].isin([0, 1, "0", "1"])].copy()
    df["result"] = df["result"].astype(int)

    if len(df) < min_rows:
        logger.info(
            "Only %d completed trades — need at least %d",
            len(df), min_rows,
        )
        return None

    return df.sort_values("timestamp").reset_index(drop=True)


def _parse_features_row(row_json: str) -> Optional[np.ndarray]:
    """Parse the features_json column into a numpy array."""
    try:
        vec = json.loads(row_json)
        arr = np.array(vec, dtype=np.float32)
        if len(arr) == N_FEATURES:
            return arr
        # Pad or truncate if schema evolved
        out = np.zeros(N_FEATURES, dtype=np.float32)
        n = min(len(arr), N_FEATURES)
        out[:n] = arr[:n]
        return out
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    df: Optional[pd.DataFrame] = None,
    test_frac: float = 0.2,
    scale: bool = True,
) -> dict:
    """
    Convert the trade log into (X_train, X_test, y_train, y_test).

    Parameters
    ----------
    df        : Pre-loaded DataFrame (from load_trades).  Loads from disk if None.
    test_frac : Fraction of most-recent trades reserved for testing.
                Time-ordered split — no shuffling (avoids data leakage).
    scale     : Apply StandardScaler to features.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test  (np.ndarray)
        scaler                             (fitted StandardScaler or None)
        feature_names                      (list[str])
        n_train, n_test                    (int)
        class_balance                      (dict)
    """
    if df is None:
        df = load_trades()
    if df is None:
        return {}

    # Parse feature vectors
    X_list, y_list = [], []
    for _, row in df.iterrows():
        vec = _parse_features_row(str(row["features_json"]))
        if vec is None:
            continue
        X_list.append(vec)
        y_list.append(int(row["result"]))

    if len(X_list) < 10:
        logger.warning("Fewer than 10 parseable feature rows — skipping dataset build")
        return {}

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)

    # Time-ordered train/test split
    n_test  = max(1, int(len(y) * test_frac))
    n_train = len(y) - n_test

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    scaler = None
    if scale:
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    wins = int(y.sum())
    losses = len(y) - wins
    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "scaler":        scaler,
        "feature_names": FEATURE_NAMES,
        "n_train":       n_train,
        "n_test":        n_test,
        "class_balance": {"wins": wins, "losses": losses, "total": len(y)},
    }


def synthetic_dataset(n: int = 1000, seed: int = 42) -> dict:
    """
    Generate a synthetic training dataset for cold-start (no live trades yet).

    Uses the backtesting engine to simulate n trades and builds the dataset
    from those simulated outcomes.  Features are randomly sampled around
    the distributions expected in real ICT setups.
    """
    from backtesting import _build_ict_candles, _run_strategy_on_window, _simulate_exit
    from ml.feature_engineering import extract
    import random

    rng = random.Random(seed)
    np.random.seed(seed)

    X_list, y_list = [], []

    # Build synthetic candle windows and extract features
    for _ in range(n):
        base_price  = rng.uniform(10_000, 70_000)
        trend       = rng.choice(["bullish", "bearish"])
        volatility  = rng.uniform(0.005, 0.025)

        raw_candles = _build_ict_candles(base_price, trend, volatility, 100)
        if raw_candles is None or len(raw_candles) < 20:
            continue

        df_candles = pd.DataFrame(raw_candles, columns=["open", "high", "low", "close"])
        signal = _run_strategy_on_window(df_candles, rr_ratio=2.0)
        if signal is None:
            continue

        outcome = _simulate_exit(signal, df_candles, rr_ratio=2.0)
        if outcome is None:
            continue

        vec = extract(df_candles, signal)
        X_list.append(vec)
        y_list.append(1 if outcome["result"] == "win" else 0)

    if len(X_list) < 20:
        logger.warning("Synthetic dataset produced only %d samples", len(X_list))
        return {}

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)

    n_test  = max(1, int(len(y) * 0.2))
    n_train = len(y) - n_test

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X[:n_train])
    X_test  = scaler.transform(X[n_train:])

    wins   = int(y.sum())
    losses = len(y) - wins
    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y[:n_train],
        "y_test":        y[n_train:],
        "scaler":        scaler,
        "feature_names": FEATURE_NAMES,
        "n_train":       n_train,
        "n_test":        n_test,
        "class_balance": {"wins": wins, "losses": losses, "total": len(y)},
        "source":        "synthetic",
    }
