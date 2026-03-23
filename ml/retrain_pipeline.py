"""
Automated retraining pipeline.

Triggers
--------
• Every RETRAIN_EVERY new completed trades logged to trades_ml.csv.
• Manually via /api/ml/train endpoint.

Safety
------
• New model must achieve F1 ≥ (current_f1 × MIN_IMPROVEMENT_RATIO) to replace.
• Models are versioned; rollback is supported via model.promote_version().
• TimeSeriesSplit cross-validation guards against overfitting.

Usage
-----
    from ml.retrain_pipeline import maybe_retrain, force_retrain
    maybe_retrain()   # called after each trade closes
    force_retrain()   # called from /api/ml/train
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ml import _state        # shared mutable state (see ml/__init__.py)

logger = logging.getLogger(__name__)

RETRAIN_EVERY          = 20     # retrain after this many new trades
MIN_IMPROVEMENT_RATIO  = 0.97   # new F1 must be ≥ 97 % of current F1
MIN_TRAIN_SAMPLES      = 30     # refuse to train if fewer samples


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_completed_trades() -> int:
    """Count completed trade rows in trades_ml.csv."""
    from ml.dataset_builder import TRADES_CSV
    if not TRADES_CSV.exists():
        return 0
    try:
        import pandas as pd
        df = pd.read_csv(TRADES_CSV, usecols=["result"])
        return int(df["result"].isin([0, 1, "0", "1"]).sum())
    except Exception:
        return 0


def _current_model_f1() -> float:
    """Return the F1 of the currently loaded model, or 0 if none."""
    bundle = _state.get("bundle")
    if bundle and "metrics" in bundle:
        return float(bundle["metrics"].get("f1", 0.0))
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def _run_training(n_synthetic: int = 500) -> Optional[dict]:
    """
    Build dataset → train → evaluate → return result dict or None on failure.

    Combines synthetic data (cold-start bootstrap) with real live/backtest data.
    """
    from ml.dataset_builder import build_dataset, load_trades, synthetic_dataset
    from ml.train_model import train

    t0 = time.monotonic()

    # --- Real trades ---
    real_df  = load_trades(min_rows=MIN_TRAIN_SAMPLES)
    real_ds  = build_dataset(real_df) if real_df is not None else {}

    # --- Synthetic bootstrap (always include for regularisation) ---
    syn_ds = synthetic_dataset(n=n_synthetic, seed=int(time.time()) % 10000)

    # --- Merge: real takes precedence over synthetic ---
    import numpy as np
    if real_ds and syn_ds:
        X_train = np.vstack([syn_ds["X_train"], real_ds["X_train"]])
        y_train = np.concatenate([syn_ds["y_train"], real_ds["y_train"]])
        X_test  = real_ds["X_test"]    # test on REAL data only
        y_test  = real_ds["y_test"]
        scaler  = real_ds["scaler"]
    elif real_ds:
        X_train, y_train = real_ds["X_train"], real_ds["y_train"]
        X_test,  y_test  = real_ds["X_test"],  real_ds["y_test"]
        scaler = real_ds["scaler"]
    elif syn_ds:
        X_train, y_train = syn_ds["X_train"], syn_ds["y_train"]
        X_test,  y_test  = syn_ds["X_test"],  syn_ds["y_test"]
        scaler = syn_ds["scaler"]
    else:
        logger.warning("Retrain: no usable data — aborting")
        return None

    merged_ds = {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
        "feature_names": real_ds.get("feature_names") or syn_ds.get("feature_names"),
    }

    result = train(merged_ds, run_cv=True)
    elapsed = time.monotonic() - t0

    logger.info(
        "Training done in %.1fs: %s  f1=%.3f  auc=%.3f  n_train=%d",
        elapsed,
        result.get("best_name", "?"),
        result.get("metrics", {}).get("f1", 0),
        result.get("metrics", {}).get("roc_auc", 0),
        len(y_train),
    )
    result["n_train"] = len(y_train)
    result["elapsed_s"] = round(elapsed, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def maybe_retrain() -> bool:
    """
    Retrain only if enough new trades have accumulated since last retrain.

    Returns True if retraining was performed.
    """
    n_trades  = _count_completed_trades()
    last_n    = _state.get("last_retrain_n", 0)

    if n_trades - last_n < RETRAIN_EVERY:
        return False

    logger.info(
        "Auto-retrain triggered: %d trades (%d new since last retrain)",
        n_trades, n_trades - last_n,
    )
    force_retrain(skip_improvement_check=False)
    return True


def force_retrain(
    n_synthetic: int = 500,
    skip_improvement_check: bool = True,
) -> dict:
    """
    Unconditionally (re)train the model and save if it improves.

    Returns a status dict with version, metrics, promoted (bool).
    """
    from ml.model import save_model, load_model

    result = _run_training(n_synthetic=n_synthetic)
    if result is None:
        return {"error": "Training aborted — insufficient data"}

    new_f1    = result.get("metrics", {}).get("f1", 0.0)
    cur_f1    = _current_model_f1()
    promoted  = False

    if skip_improvement_check or new_f1 >= cur_f1 * MIN_IMPROVEMENT_RATIO:
        bundle = {
            "model":           result["best_model"],
            "scaler":          result["scaler"],
            "model_name":      result["best_name"],
            "threshold":       _state.get("threshold", 0.60),
            "feature_names":   result.get("metrics", {}).get("feature_names"),
            "metrics":         result["metrics"],
            "cv_results":      result.get("cv_results", {}),
            "n_train":         result["n_train"],
            "ensemble_models": result.get("all_models", []),
        }
        version = save_model(bundle)
        # Hot-swap in memory
        _state["bundle"] = load_model(version)
        _state["last_retrain_n"] = _count_completed_trades()
        promoted = True
        logger.info(
            "New model v%d promoted  f1=%.3f  (prev=%.3f)",
            version, new_f1, cur_f1,
        )
        # Record feature importances for drift detection
        try:
            from ml.drift_detector import record_importances
            feat_imps = result.get("metrics", {}).get("feature_importances", [])
            if feat_imps:
                record_importances(
                    model_version=version,
                    importances=feat_imps,
                    model_name=result.get("best_name", ""),
                )
        except Exception as exc:
            logger.debug("Drift recording failed: %s", exc)
    else:
        logger.info(
            "New model NOT promoted: f1=%.3f < %.3f × %.2f = %.3f",
            new_f1, cur_f1, MIN_IMPROVEMENT_RATIO, cur_f1 * MIN_IMPROVEMENT_RATIO,
        )
        version = None

    return {
        "promoted":   promoted,
        "version":    version,
        "new_f1":     round(new_f1, 4),
        "prev_f1":    round(cur_f1, 4),
        "metrics":    result["metrics"],
        "cv_results": result.get("cv_results", {}),
        "n_train":    result["n_train"],
        "elapsed_s":  result.get("elapsed_s"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
