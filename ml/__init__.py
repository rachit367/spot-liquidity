"""
ML module public API.

Singleton access
----------------
    from ml import get_model, score, record_outcome

    bundle   = get_model()          # loads or trains on first call
    decision = score(bundle, df, signal)
    # … trade executes …
    record_outcome(trade_id, actual_result)

``_state`` dict is shared across all submodules so the loaded bundle and
counters stay consistent within one server process.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from execution import TradeSignal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Shared mutable state (process-level singleton)
# ─────────────────────────────────────────────────────────────────────────────

_state: dict = {
    "bundle":          None,   # loaded model bundle dict
    "threshold":       0.60,   # default confidence threshold
    "last_retrain_n":  0,      # n_trades at time of last retrain
    "pending_trades":  {},     # {trade_id: {features, confidence}} awaiting outcome
    "perf_tracker":    None,   # PerformanceTracker instance
}


# ─────────────────────────────────────────────────────────────────────────────
# get_model — lazy initialiser
# ─────────────────────────────────────────────────────────────────────────────

def get_model() -> dict:
    """
    Return the active model bundle (dict).

    On first call:
      1. Try to load the latest saved model from disk.
      2. If none exists, train from synthetic backtest data and save.

    Subsequent calls return the in-memory bundle directly (no disk I/O).
    """
    if _state["bundle"] is not None:
        return _state["bundle"]

    # Try to load from disk
    from ml.model import load_model
    bundle = load_model()
    if bundle is not None:
        _state["bundle"] = bundle
        logger.info(
            "Loaded ML model v%d (%s)  f1=%.3f",
            bundle.get("version", "?"),
            bundle.get("model_name", "?"),
            bundle.get("metrics", {}).get("f1", 0),
        )
        return bundle

    # Cold-start: train on synthetic data
    logger.info("No saved ML model found — performing cold-start training …")
    from ml.retrain_pipeline import force_retrain
    result = force_retrain(n_synthetic=500, skip_improvement_check=True)
    if "error" in result:
        logger.error("Cold-start training failed: %s", result["error"])
        # Return an empty bundle so the rest of the system can run without ML
        _state["bundle"] = {"model": None, "scaler": None, "model_name": "none",
                            "threshold": _state["threshold"], "version": 0}
        return _state["bundle"]

    return _state["bundle"]


# ─────────────────────────────────────────────────────────────────────────────
# score — wrapper around inference.score_signal
# ─────────────────────────────────────────────────────────────────────────────

def score(
    df: pd.DataFrame,
    signal: TradeSignal,
    timestamp: Optional[datetime] = None,
    threshold: Optional[float] = None,
):
    """
    Score a signal and return a Decision.

    ``score`` automatically calls get_model() so callers don't need to
    manage the bundle themselves.
    """
    from ml.inference import score_signal
    bundle = get_model()
    return score_signal(
        model_bundle=bundle,
        df=df,
        signal=signal,
        timestamp=timestamp,
        threshold=threshold or _state["threshold"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# record_outcome — online learning feedback
# ─────────────────────────────────────────────────────────────────────────────

def record_outcome(
    trade_id: str,
    actual_result: int,     # 1 = win, 0 = loss
    entry_price: float = 0,
    exit_price: float = 0,
    pnl: float = 0,
    **extra_fields,
) -> None:
    """
    Called when a trade closes (SL or TP hit).

    1. Pops the pending_trades entry.
    2. Writes the completed trade to trades_ml.csv.
    3. Updates the PerformanceTracker.
    4. Triggers maybe_retrain() if enough new trades accumulated.
    """
    from ml.dataset_builder import append_trade
    from ml.retrain_pipeline import maybe_retrain

    pending = _state["pending_trades"].pop(trade_id, {})

    # Build trade record
    record = {
        "trade_id":         trade_id,
        "timestamp":        pending.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "symbol":           pending.get("symbol", ""),
        "direction":        pending.get("direction", ""),
        "entry_price":      entry_price or pending.get("entry", 0),
        "exit_price":       exit_price,
        "sl":               pending.get("sl", 0),
        "tp":               pending.get("tp", 0),
        "quantity":         pending.get("quantity", 0),
        "pnl":              pnl,
        "pnl_pct":          extra_fields.get("pnl_pct", 0),
        "rr_achieved":      extra_fields.get("rr_achieved", 0),
        "result":           actual_result,
        "strategy":         pending.get("strategy", "ICT"),
        "duration_bars":    extra_fields.get("duration_bars", 0),
        "model_confidence": pending.get("confidence", None),
        "model_prediction": pending.get("prediction", None),
        "features":         pending.get("features"),
    }
    append_trade(record)

    # Update performance tracker
    tracker = _get_tracker()
    tracker.record(
        trade_id=trade_id,
        confidence=float(pending.get("confidence", 0.5)),
        predicted=int(pending.get("prediction", 1)),
        actual=actual_result,
        features=pending.get("features"),
    )

    logger.info("Trade %s closed: result=%d  conf=%.2f", trade_id, actual_result,
                pending.get("confidence", -1))

    # Maybe retrain
    maybe_retrain()


def register_pending(
    trade_id: str,
    signal: TradeSignal,
    features: np.ndarray,
    confidence: float,
    prediction: int,
    timestamp: Optional[datetime] = None,
) -> None:
    """
    Store the feature snapshot and prediction for an open trade.
    Called at execution time so outcome can be paired later.
    """
    _state["pending_trades"][trade_id] = {
        "timestamp":  (timestamp or datetime.now(timezone.utc)).isoformat(),
        "symbol":     signal.symbol,
        "direction":  signal.direction,
        "entry":      signal.entry,
        "sl":         signal.stop_loss,
        "tp":         signal.take_profit,
        "features":   features,
        "confidence": confidence,
        "prediction": prediction,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_tracker():
    from ml.inference import get_tracker
    if _state["perf_tracker"] is None:
        _state["perf_tracker"] = get_tracker()
    return _state["perf_tracker"]


def get_status() -> dict:
    """Return a dict describing the current model state (for /api/ml/status)."""
    bundle  = _state.get("bundle")
    tracker = _get_tracker()

    if bundle is None:
        return {"status": "not_loaded"}

    return {
        "status":         "ready" if bundle.get("model") else "no_model",
        "model_name":     bundle.get("model_name", "none"),
        "version":        bundle.get("version", 0),
        "trained_at":     bundle.get("trained_at", ""),
        "threshold":      _state["threshold"],
        "f1":             bundle.get("metrics", {}).get("f1"),
        "roc_auc":        bundle.get("metrics", {}).get("roc_auc"),
        "n_train":        bundle.get("n_train", 0),
        "n_pending":      len(_state["pending_trades"]),
        "performance":    tracker.summary(),
    }


def set_threshold(value: float) -> None:
    """Update the confidence threshold (also persists to in-memory bundle)."""
    value = max(0.5, min(0.95, value))
    _state["threshold"] = value
    if _state["bundle"]:
        _state["bundle"]["threshold"] = value
    logger.info("ML threshold set to %.2f", value)
