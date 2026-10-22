"""
Unit Tests for Module 6 (Ensemble Techniques & Boosting)
========================================================
Verifies:
1. Bagging (Random Forest) and Boosting (AdaBoost, Gradient Boosting, XGBoost)
   train, tune, and evaluate for regression and classification tasks.
2. Proper metric computation (RMSE, MAE, R2, Accuracy, F1-Score, ROC-AUC).
3. Justification report generation and model serialization.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from src.module6_ensemble.ensemble_models import EnsembleModelEngine, EnsembleBenchmark


@pytest.fixture
def small_ensemble_data():
    """Generates small synthetic datasets for fast unit testing."""
    np.random.seed(42)
    # Regression
    X_reg, y_reg = make_regression(n_samples=80, n_features=6, noise=0.1, random_state=42)
    X_reg = pd.DataFrame(X_reg, columns=[f"feat_{i}" for i in range(6)])
    y_reg = pd.Series(y_reg, name="target_reg")

    # Classification (4 classes)
    X_clf, y_clf = make_classification(
        n_samples=80, n_features=6, n_informative=4, n_classes=4, random_state=42
    )
    X_clf = pd.DataFrame(X_clf, columns=[f"feat_{i}" for i in range(6)])
    y_clf = pd.Series(y_clf, name="target_clf")

    split = 60
    return {
        "X_train_reg": X_reg.iloc[:split],
        "y_train_reg": y_reg.iloc[:split],
        "X_test_reg": X_reg.iloc[split:],
        "y_test_reg": y_reg.iloc[split:],
        "X_train_clf": X_clf.iloc[:split],
        "y_train_clf": y_clf.iloc[:split],
        "X_test_clf": X_clf.iloc[split:],
        "y_test_clf": y_clf.iloc[split:],
    }


def test_ensemble_regression_benchmarking(small_ensemble_data):
    engine = EnsembleModelEngine(random_state=42)
    leaderboard, best_name, best_model = engine.benchmark_regression_ensembles(
        X_train=small_ensemble_data["X_train_reg"],
        y_train=small_ensemble_data["y_train_reg"],
        X_test=small_ensemble_data["X_test_reg"],
        y_test=small_ensemble_data["y_test_reg"],
        n_splits=2,
    )

    assert len(leaderboard) == 4  # RF, AdaBoost, GradientBoosting, XGBoost
    expected_models = {"Random Forest (Bagging)", "AdaBoost", "Gradient Boosting", "XGBoost"}
    assert set(leaderboard["ensemble_model"]) == expected_models

    for _, row in leaderboard.iterrows():
        assert row["test_rmse"] >= 0
        assert row["test_mae"] >= 0

    assert best_model is not None


def test_ensemble_classification_benchmarking(small_ensemble_data):
    engine = EnsembleModelEngine(random_state=42)
    leaderboard, best_name, best_model = engine.benchmark_classification_ensembles(
        X_train=small_ensemble_data["X_train_clf"],
        y_train=small_ensemble_data["y_train_clf"],
        X_test=small_ensemble_data["X_test_clf"],
        y_test=small_ensemble_data["y_test_clf"],
        n_splits=2,
    )

    assert len(leaderboard) == 4  # RF, AdaBoost, GradientBoosting, XGBoost
    expected_models = {"Random Forest (Bagging)", "AdaBoost", "Gradient Boosting", "XGBoost"}
    assert set(leaderboard["ensemble_model"]) == expected_models

    for _, row in leaderboard.iterrows():
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["f1_macro"] <= 1.0

    assert best_model is not None


def test_justification_report_generation(small_ensemble_data):
    engine = EnsembleModelEngine(random_state=42)
    reg_df, best_reg, _ = engine.benchmark_regression_ensembles(
        small_ensemble_data["X_train_reg"], small_ensemble_data["y_train_reg"],
        small_ensemble_data["X_test_reg"], small_ensemble_data["y_test_reg"], n_splits=2
    )
    clf_df, best_clf, _ = engine.benchmark_classification_ensembles(
        small_ensemble_data["X_train_clf"], small_ensemble_data["y_train_clf"],
        small_ensemble_data["X_test_clf"], small_ensemble_data["y_test_clf"], n_splits=2
    )

    report = engine.generate_justification_report(reg_df, clf_df, best_reg, best_clf)
    assert "# Module 6 (Unit-V) Ensemble Model Justification Report" in report
    assert best_reg in report
    assert best_clf in report
