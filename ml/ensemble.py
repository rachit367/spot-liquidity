"""
Ensemble voting — all trained classifiers vote on each signal.

Instead of using only the single best model, the ensemble runs all
available classifiers and requires a majority (2/3) or unanimous (3/3)
agreement to approve a trade. This significantly reduces false positives.

Usage
-----
    from ml.ensemble import EnsembleVoter
    voter = EnsembleVoter(models, scaler)
    result = voter.vote(X)
    if result["approved"]:
        executor.execute(signal)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EnsembleVoter:
    """
    Ensemble classifier that requires multiple models to agree.

    Modes
    -----
    - "majority"  : ≥ 50% of models must predict win (default)
    - "unanimous" : 100% of models must predict win
    """

    def __init__(
        self,
        models: list[dict],      # [{name, model, metrics}, ...]
        scaler: Any = None,
        mode: str = "majority",
        threshold: float = 0.60,
    ) -> None:
        self.models = [m for m in models if m.get("model") is not None]
        self.scaler = scaler
        self.mode = mode
        self.threshold = threshold

    @property
    def n_models(self) -> int:
        return len(self.models)

    def vote(self, X: np.ndarray) -> dict:
        """
        Run all models on the input and return the voting result.

        Parameters
        ----------
        X : Feature array, shape (1, n_features). Already scaled.

        Returns
        -------
        dict with:
            approved       — bool, whether the ensemble approves
            confidence     — float, average confidence across models
            votes_for      — int, models predicting win
            votes_against  — int, models predicting loss
            n_models       — int, total models in ensemble
            mode           — str, voting mode used
            details        — list of per-model results
            reason         — str, human-readable explanation
        """
        if not self.models:
            return {
                "approved": False,
                "confidence": 0.5,
                "votes_for": 0,
                "votes_against": 0,
                "n_models": 0,
                "mode": self.mode,
                "details": [],
                "reason": "No models available in ensemble",
            }

        details = []
        votes_for = 0
        confidences = []

        for entry in self.models:
            name = entry["name"]
            model = entry["model"]

            try:
                proba = float(model.predict_proba(X)[0, 1])
            except Exception as exc:
                logger.warning("Ensemble: %s predict failed: %s", name, exc)
                proba = 0.5

            vote = proba >= self.threshold
            if vote:
                votes_for += 1
            confidences.append(proba)

            details.append({
                "name": name,
                "confidence": round(proba, 4),
                "vote": "approve" if vote else "reject",
                "f1": round(entry.get("metrics", {}).get("f1", 0), 4),
            })

        votes_against = len(self.models) - votes_for
        avg_confidence = float(np.mean(confidences))

        # Determine approval based on mode
        if self.mode == "unanimous":
            approved = votes_for == len(self.models)
        else:  # majority
            approved = votes_for > len(self.models) / 2

        # Build reason
        vote_str = " | ".join(
            f"{d['name']}={'✓' if d['vote'] == 'approve' else '✗'}({d['confidence']:.2f})"
            for d in details
        )
        reason = (
            f"Ensemble ({self.mode}): {votes_for}/{len(self.models)} approve "
            f"| avg_conf={avg_confidence:.2f} | {vote_str}"
        )

        if approved:
            logger.info("Ensemble APPROVED: %s", reason)
        else:
            logger.info("Ensemble REJECTED: %s", reason)

        return {
            "approved": approved,
            "confidence": round(avg_confidence, 4),
            "votes_for": votes_for,
            "votes_against": votes_against,
            "n_models": len(self.models),
            "mode": self.mode,
            "details": details,
            "reason": reason,
        }
