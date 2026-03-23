"""
Tests for the ML training module.
"""

import numpy as np
import pytest

from ml.train_model import _get_classifiers, train, evaluate, cross_validate


@pytest.fixture
def synthetic_data():
    """Generate simple synthetic data for testing training."""
    np.random.seed(42)
    n = 200
    n_features = 26

    X = np.random.randn(n, n_features)
    # Simple separable pattern: positive sum of first 3 features → win
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)

    n_test = 40
    return {
        "X_train": X[:-n_test],
        "X_test":  X[-n_test:],
        "y_train": y[:-n_test],
        "y_test":  y[-n_test:],
        "feature_names": [f"feat_{i}" for i in range(n_features)],
    }


class TestGetClassifiers:
    """Tests for _get_classifiers."""

    def test_always_has_random_forest(self):
        clfs = _get_classifiers()
        assert "random_forest" in clfs

    def test_returns_dict(self):
        clfs = _get_classifiers()
        assert isinstance(clfs, dict)
        assert len(clfs) >= 1


class TestTrain:
    """Tests for the train function."""

    def test_returns_best_model(self, synthetic_data):
        result = train(synthetic_data, run_cv=False)
        assert result["best_model"] is not None
        assert result["best_name"] is not None
        assert "metrics" in result

    def test_metrics_are_reasonable(self, synthetic_data):
        result = train(synthetic_data, run_cv=False)
        metrics = result["metrics"]
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_cross_validation_runs(self, synthetic_data):
        result = train(synthetic_data, run_cv=True)
        if result.get("cv_results"):
            assert "cv_f1_mean" in result["cv_results"]
            assert result["cv_results"]["cv_f1_mean"] >= 0


class TestEvaluate:
    """Tests for the evaluate function."""

    def test_evaluation_metrics(self, synthetic_data):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        result = evaluate(
            clf,
            synthetic_data["X_test"],
            synthetic_data["y_test"],
            feature_names=synthetic_data["feature_names"],
        )

        assert "accuracy" in result
        assert "f1" in result
        assert "confusion_matrix" in result
        assert "feature_importances" in result
        assert len(result["feature_importances"]) > 0

    def test_feature_importances_sorted(self, synthetic_data):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(synthetic_data["X_train"], synthetic_data["y_train"])

        result = evaluate(
            clf,
            synthetic_data["X_test"],
            synthetic_data["y_test"],
            feature_names=synthetic_data["feature_names"],
        )

        imps = result["feature_importances"]
        # Should be sorted descending
        values = [f["importance"] for f in imps]
        assert values == sorted(values, reverse=True)


class TestCrossValidate:
    """Tests for cross_validate function."""

    def test_returns_cv_scores(self, synthetic_data):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)

        X = np.vstack([synthetic_data["X_train"], synthetic_data["X_test"]])
        y = np.concatenate([synthetic_data["y_train"], synthetic_data["y_test"]])

        result = cross_validate(clf, X, y, n_folds=3)
        assert "cv_f1_mean" in result
        assert "cv_accuracy_mean" in result
        assert result["cv_f1_mean"] >= 0
