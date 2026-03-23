"""
Inference layer — probability-based trade filtering.

Only passes trades through when model confidence exceeds a threshold.
Also suggests an adjusted R:R ratio based on confidence level.

Usage
-----
    from ml.inference import score_signal, Decision
    decision = score_signal(model_bundle, df, signal)
    if decision.approved:
        executor.execute(signal)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from execution import TradeSignal
from ml.feature_engineering import extract, features_to_dict, FEATURE_NAMES

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.60    # minimum win-probability to approve a trade
RR_BOOST_HIGH     = 2.5     # use higher R:R when confidence ≥ 0.75
RR_BOOST_LOW      = 2.0     # default R:R


@dataclass
class Decision:
    """Result of the ML inference step."""
    approved:        bool
    confidence:      float          # predicted win probability (0–1)
    threshold:       float
    suggested_rr:    float
    model_name:      str
    feature_vec:     np.ndarray = field(repr=False)
    top_features:    list[dict]  = field(default_factory=list)
    reason:          str         = ""


def score_signal(
    model_bundle: dict,
    df: pd.DataFrame,
    signal: TradeSignal,
    timestamp: Optional[datetime] = None,
    threshold: Optional[float] = None,
) -> Decision:
    """
    Score a signal using the loaded model.

    Parameters
    ----------
    model_bundle : dict returned by model.load_model() or ml.__init__.get_model()
                   Keys: model, scaler, threshold, model_name, feature_names
    df           : OHLCV DataFrame used to generate the signal
    signal       : The TradeSignal to score
    timestamp    : UTC datetime of the signal (uses now() if None)
    threshold    : Override the bundle's default threshold

    Returns
    -------
    Decision with approved=True if confidence ≥ threshold.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    if threshold is None:
        # Try dynamic threshold first
        try:
            from ml.dynamic_threshold import get_dynamic_threshold
            dt = get_dynamic_threshold()
            if dt.enabled:
                threshold = dt.get_threshold()
            else:
                threshold = float(model_bundle.get("threshold", DEFAULT_THRESHOLD))
        except Exception:
            threshold = float(model_bundle.get("threshold", DEFAULT_THRESHOLD))

    clf    = model_bundle.get("model")
    scaler = model_bundle.get("scaler")

    # Extract features
    vec = extract(df, signal, timestamp=timestamp)

    # Scale
    X = vec.reshape(1, -1)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as exc:
            logger.warning("Scaler transform failed: %s — using raw features", exc)

    # ── Ensemble voting (if multiple models available) ────────────────
    ensemble_models = model_bundle.get("ensemble_models", [])
    if len(ensemble_models) >= 2:
        try:
            from ml.ensemble import EnsembleVoter
            voter = EnsembleVoter(
                models=ensemble_models,
                scaler=None,   # already scaled above
                mode="majority",
                threshold=threshold,
            )
            vote_result = voter.vote(X)

            # Extract top features from best model
            top_features: list[dict] = []
            if clf is not None and hasattr(clf, "feature_importances_"):
                imps  = clf.feature_importances_
                names = FEATURE_NAMES
                pairs = sorted(zip(names, vec.tolist()), key=lambda p: -imps[names.index(p[0])])
                top_features = [
                    {"name": n, "value": round(float(v), 4), "importance": round(float(imps[names.index(n)]), 4)}
                    for n, v in pairs[:5]
                ]

            model_name = f"ensemble({vote_result['n_models']})"
            suggested_rr = RR_BOOST_HIGH if vote_result["confidence"] >= 0.75 else RR_BOOST_LOW

            return Decision(
                approved=vote_result["approved"],
                confidence=vote_result["confidence"],
                threshold=threshold,
                suggested_rr=suggested_rr,
                model_name=model_name,
                feature_vec=vec,
                top_features=top_features,
                reason=vote_result["reason"],
            )
        except Exception as exc:
            logger.warning("Ensemble voting failed, falling back to single model: %s", exc)

    # ── Single model fallback ────────────────────────────────────────
    confidence = 0.5
    model_name = model_bundle.get("model_name", "unknown")
    if clf is not None:
        try:
            confidence = float(clf.predict_proba(X)[0, 1])
        except Exception as exc:
            logger.warning("Model predict_proba failed: %s — defaulting to 0.5", exc)
    else:
        logger.debug("No model loaded — score_signal returning neutral confidence")

    # Adjusted R:R suggestion
    suggested_rr = RR_BOOST_HIGH if confidence >= 0.75 else RR_BOOST_LOW

    # Top 5 most important features
    top_features: list[dict] = []
    if clf is not None and hasattr(clf, "feature_importances_"):
        imps  = clf.feature_importances_
        names = FEATURE_NAMES
        pairs = sorted(zip(names, vec.tolist()), key=lambda p: -imps[names.index(p[0])])
        top_features = [
            {"name": n, "value": round(float(v), 4), "importance": round(float(imps[names.index(n)]), 4)}
            for n, v in pairs[:5]
        ]

    approved = confidence >= threshold

    reason = (
        f"Model {model_name} | confidence={confidence:.2f} "
        f"{'≥' if approved else '<'} threshold={threshold:.2f}"
    )

    if approved:
        logger.info("ML APPROVED: %s  conf=%.2f  rr=%.1f", signal.symbol, confidence, suggested_rr)
    else:
        logger.info("ML REJECTED: %s  conf=%.2f  below threshold=%.2f", signal.symbol, confidence, threshold)

    return Decision(
        approved=approved,
        confidence=confidence,
        threshold=threshold,
        suggested_rr=suggested_rr,
        model_name=model_name,
        feature_vec=vec,
        top_features=top_features,
        reason=reason,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model performance tracker
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceTracker:
    """
    Tracks prediction vs actual outcome to identify failure patterns.

    Stores a rolling window of (confidence, prediction, actual) tuples.
    """

    def __init__(self, window: int = 200) -> None:
        self._window = window
        self._records: list[dict] = []

    def record(
        self,
        trade_id: str,
        confidence: float,
        predicted: int,
        actual: int,
        features: Optional[np.ndarray] = None,
    ) -> None:
        self._records.append({
            "trade_id":   trade_id,
            "confidence": confidence,
            "predicted":  predicted,
            "actual":     actual,
            "correct":    int(predicted == actual),
            "features":   features.tolist() if features is not None else None,
        })
        if len(self._records) > self._window:
            self._records.pop(0)

    def summary(self) -> dict:
        """
        Return accuracy, calibration by confidence bucket, and failure patterns.
        """
        if not self._records:
            return {"n": 0, "accuracy": None}

        n       = len(self._records)
        correct = sum(r["correct"] for r in self._records)

        # Calibration: bucket by confidence
        buckets: dict[str, list] = {"low": [], "mid": [], "high": []}
        for r in self._records:
            c = r["confidence"]
            if c < 0.55:
                buckets["low"].append(r["correct"])
            elif c < 0.70:
                buckets["mid"].append(r["correct"])
            else:
                buckets["high"].append(r["correct"])

        calibration = {}
        for bucket, vals in buckets.items():
            if vals:
                calibration[bucket] = {
                    "n":        len(vals),
                    "accuracy": round(sum(vals) / len(vals), 3),
                }

        # Recent trend (last 20)
        recent = self._records[-20:]
        recent_acc = sum(r["correct"] for r in recent) / len(recent) if recent else None

        return {
            "n":           n,
            "accuracy":    round(correct / n, 3),
            "recent_acc":  round(recent_acc, 3) if recent_acc is not None else None,
            "calibration": calibration,
        }


# Module-level singleton
_tracker = PerformanceTracker()


def get_tracker() -> PerformanceTracker:
    return _tracker
