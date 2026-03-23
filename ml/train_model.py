"""
Model training — RandomForest + XGBoost classifiers.

Evaluation metrics: accuracy, precision, recall, F1, confusion matrix,
                    feature importances, ROC-AUC.

Usage
-----
    from ml.train_model import train, evaluate
    result = train(dataset)   # dataset from dataset_builder.build_dataset()
    print(result["report"])
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports — sklearn / xgboost may not be installed
# ─────────────────────────────────────────────────────────────────────────────

def _sklearn():
    try:
        import sklearn
        return sklearn
    except ImportError:
        raise ImportError(
            "scikit-learn is required for ML features. "
            "Run: pip install scikit-learn xgboost"
        )


def _get_classifiers() -> dict[str, Any]:
    """Return the candidate classifiers (RF, XGBoost, LightGBM)."""
    from sklearn.ensemble import RandomForestClassifier
    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier
        classifiers["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=1,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        logger.info("xgboost not installed — skipping")
    try:
        from lightgbm import LGBMClassifier
        classifiers["lightgbm"] = LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        logger.info("lightgbm not installed — skipping")
    return classifiers


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray,
             feature_names: list[str] | None = None) -> dict:
    """
    Compute classification metrics for a fitted model on the test set.

    Returns a dict with:
        accuracy, precision, recall, f1, roc_auc,
        confusion_matrix (2×2 list),
        feature_importances (list[{name, importance}] sorted desc),
        report (human-readable string),
        n_test, n_wins, n_losses.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score,
    )

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    f1   = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_test, y_pred).tolist()

    # Feature importances (if supported)
    feat_imp = []
    if feature_names and hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        pairs = sorted(zip(feature_names, imps), key=lambda x: -x[1])
        feat_imp = [{"name": n, "importance": round(float(v), 4)} for n, v in pairs]

    report = (
        f"  Accuracy  : {acc:.3f}\n"
        f"  Precision : {prec:.3f}\n"
        f"  Recall    : {rec:.3f}\n"
        f"  F1        : {f1:.3f}\n"
        f"  ROC-AUC   : {auc:.3f}\n"
        f"  Confusion : TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}\n"
        f"  n_test    : {len(y_test)}  (wins={int(y_test.sum())})"
    )

    return {
        "accuracy":            acc,
        "precision":           prec,
        "recall":              rec,
        "f1":                  f1,
        "roc_auc":             auc,
        "confusion_matrix":    cm,
        "feature_importances": feat_imp,
        "report":              report,
        "n_test":              len(y_test),
        "n_wins":              int(y_test.sum()),
        "n_losses":            int(len(y_test) - y_test.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation helper
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(model_cls, X: np.ndarray, y: np.ndarray,
                   n_folds: int = 5) -> dict:
    """
    Time-series aware cross-validation using TimeSeriesSplit.

    Returns mean ± std for accuracy, f1, roc_auc.
    """
    from sklearn.model_selection import TimeSeriesSplit, cross_validate as skl_cv
    from sklearn.base import clone

    tscv  = TimeSeriesSplit(n_splits=n_folds)
    model = clone(model_cls)
    scores = skl_cv(
        model, X, y,
        cv=tscv,
        scoring=["accuracy", "f1", "roc_auc"],
        error_score=0.0,
    )
    return {
        "cv_accuracy_mean":  float(scores["test_accuracy"].mean()),
        "cv_accuracy_std":   float(scores["test_accuracy"].std()),
        "cv_f1_mean":        float(scores["test_f1"].mean()),
        "cv_f1_std":         float(scores["test_f1"].std()),
        "cv_roc_auc_mean":   float(scores["test_roc_auc"].mean()),
        "cv_roc_auc_std":    float(scores["test_roc_auc"].std()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(dataset: dict, run_cv: bool = True) -> dict:
    """
    Train all candidate classifiers on the dataset and return the best one,
    plus all fitted models for ensemble voting.

    Parameters
    ----------
    dataset  : Output of dataset_builder.build_dataset() or synthetic_dataset().
    run_cv   : Whether to run TimeSeriesSplit cross-validation (5-fold).

    Returns
    -------
    dict with:
        best_model    : fitted sklearn/xgboost model
        best_name     : "random_forest", "xgboost", or "lightgbm"
        scaler        : fitted StandardScaler (pass-through from dataset)
        metrics       : evaluation dict from evaluate()
        cv_results    : cross-validation scores (if run_cv=True)
        all_metrics   : {model_name: metrics_dict} for all models
        all_models    : [{name, model, metrics}, ...] for ensemble voting
    """
    _sklearn()   # ensure installed

    X_train = dataset["X_train"]
    X_test  = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test  = dataset["y_test"]
    feat_names = dataset.get("feature_names")

    classifiers = _get_classifiers()
    best_model  = None
    best_name   = None
    best_f1     = -1.0
    all_metrics = {}
    all_models  = []

    for name, clf in classifiers.items():
        logger.info("Training %s on %d samples …", name, len(y_train))
        clf.fit(X_train, y_train)
        m = evaluate(clf, X_test, y_test, feature_names=feat_names)
        all_metrics[name] = m
        all_models.append({"name": name, "model": clf, "metrics": m})
        logger.info(
            "%s  acc=%.3f  f1=%.3f  auc=%.3f",
            name, m["accuracy"], m["f1"], m["roc_auc"],
        )
        if m["f1"] > best_f1:
            best_f1   = m["f1"]
            best_model = clf
            best_name  = name

    cv_results = {}
    if run_cv and best_model is not None:
        try:
            X_all = np.vstack([X_train, X_test])
            y_all = np.concatenate([y_train, y_test])
            cv_results = cross_validate(best_model, X_all, y_all)
            logger.info(
                "CV  acc=%.3f±%.3f  f1=%.3f±%.3f",
                cv_results["cv_accuracy_mean"], cv_results["cv_accuracy_std"],
                cv_results["cv_f1_mean"],       cv_results["cv_f1_std"],
            )
        except Exception as exc:
            logger.warning("Cross-validation failed: %s", exc)

    return {
        "best_model":  best_model,
        "best_name":   best_name,
        "scaler":      dataset.get("scaler"),
        "metrics":     all_metrics.get(best_name, {}),
        "cv_results":  cv_results,
        "all_metrics": all_metrics,
        "all_models":  all_models,
    }

