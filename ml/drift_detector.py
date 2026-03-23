"""
Feature importance drift detector — tracks how feature importances
change across model retrains and alerts on regime changes.

Stores importances in a JSON log and computes drift scores to detect
when the market regime has fundamentally shifted.

Usage
-----
    from ml.drift_detector import record_importances, get_drift_report
    record_importances(model_version=3, importances=[...])
    report = get_drift_report()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DRIFT_LOG = Path(__file__).resolve().parent.parent / "logs" / "drift_log.json"
DRIFT_THRESHOLD = 0.50   # 50% change = "drifted"
REGIME_TOP_N    = 3      # if top-N features reorder → regime change


# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────

def _load_log() -> list[dict]:
    """Load the drift log from disk."""
    if not DRIFT_LOG.exists():
        return []
    try:
        with open(DRIFT_LOG, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_log(entries: list[dict]) -> None:
    """Save the drift log to disk."""
    DRIFT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_LOG, "w") as f:
        json.dump(entries, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Record importances after each retrain
# ─────────────────────────────────────────────────────────────────────────────

def record_importances(
    model_version: int,
    importances: list[dict],   # [{name, importance}, ...]
    model_name: str = "",
) -> None:
    """
    Record feature importances from a retrained model.

    Parameters
    ----------
    model_version : Version number of the model
    importances   : List of {name: str, importance: float} sorted by importance desc
    model_name    : Name of the model (e.g. "xgboost")
    """
    entries = _load_log()
    entries.append({
        "version":      model_version,
        "model_name":   model_name,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "importances":  importances,
    })

    # Keep last 50 entries
    if len(entries) > 50:
        entries = entries[-50:]

    _save_log(entries)
    logger.info("Drift detector: recorded importances for v%d", model_version)


# ─────────────────────────────────────────────────────────────────────────────
# Drift analysis
# ─────────────────────────────────────────────────────────────────────────────

def _compute_drift(prev: list[dict], curr: list[dict]) -> list[dict]:
    """
    Compare two sets of feature importances and compute drift scores.

    Returns a list of {feature, prev_imp, curr_imp, drift_score, drifted} dicts.
    """
    prev_map = {f["name"]: f["importance"] for f in prev}
    curr_map = {f["name"]: f["importance"] for f in curr}

    all_features = set(prev_map.keys()) | set(curr_map.keys())
    results = []

    for feat in all_features:
        prev_val = prev_map.get(feat, 0.0)
        curr_val = curr_map.get(feat, 0.0)
        denom = max(abs(prev_val), 1e-6)
        drift_score = abs(curr_val - prev_val) / denom

        results.append({
            "feature":     feat,
            "prev_imp":    round(prev_val, 4),
            "curr_imp":    round(curr_val, 4),
            "drift_score": round(drift_score, 4),
            "drifted":     drift_score > DRIFT_THRESHOLD,
        })

    results.sort(key=lambda x: -x["drift_score"])
    return results


def _check_regime_change(prev: list[dict], curr: list[dict]) -> dict:
    """
    Check if the top-N features have significantly reordered.
    """
    prev_top = [f["name"] for f in prev[:REGIME_TOP_N]]
    curr_top = [f["name"] for f in curr[:REGIME_TOP_N]]

    # Check if the set of top features changed
    set_changed = set(prev_top) != set(curr_top)

    # Check if the order changed
    order_changed = prev_top != curr_top

    return {
        "regime_change":   set_changed,
        "order_changed":   order_changed,
        "prev_top":        prev_top,
        "curr_top":        curr_top,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public report
# ─────────────────────────────────────────────────────────────────────────────

def get_drift_report() -> dict:
    """
    Generate a drift analysis report comparing the last two model versions.

    Returns
    -------
    dict with:
        has_drift        — bool, True if any feature drifted significantly
        regime_change    — bool, True if top features reordered
        drifted_features — list of features with > 50% importance change
        all_drift        — full drift analysis per feature
        prev_version     — version number of the previous model
        curr_version     — version number of the current model
        alerts           — list of human-readable alert strings
    """
    entries = _load_log()

    if len(entries) < 2:
        return {
            "error": "Need at least 2 model versions to detect drift",
            "versions_recorded": len(entries),
        }

    prev = entries[-2]
    curr = entries[-1]

    drift = _compute_drift(prev["importances"], curr["importances"])
    regime = _check_regime_change(prev["importances"], curr["importances"])

    drifted = [d for d in drift if d["drifted"]]
    has_drift = len(drifted) > 0

    # Build alerts
    alerts = []
    if regime["regime_change"]:
        alerts.append(
            f"⚠ REGIME CHANGE: Top features changed from "
            f"{regime['prev_top']} → {regime['curr_top']}"
        )
    if regime["order_changed"] and not regime["regime_change"]:
        alerts.append(
            f"📊 Feature reorder: {regime['prev_top']} → {regime['curr_top']}"
        )
    for d in drifted[:3]:
        direction = "↑" if d["curr_imp"] > d["prev_imp"] else "↓"
        alerts.append(
            f"{direction} {d['feature']}: {d['prev_imp']:.3f} → {d['curr_imp']:.3f} "
            f"({d['drift_score']*100:.0f}% drift)"
        )

    return {
        "has_drift":         has_drift,
        "regime_change":     regime["regime_change"],
        "order_changed":     regime["order_changed"],
        "drifted_features":  drifted,
        "all_drift":         drift[:15],   # top 15
        "prev_version":      prev["version"],
        "curr_version":      curr["version"],
        "prev_top_features": regime["prev_top"],
        "curr_top_features": regime["curr_top"],
        "alerts":            alerts,
        "versions_recorded": len(entries),
    }
