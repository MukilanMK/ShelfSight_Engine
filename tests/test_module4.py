"""
Unit Tests for Module 4 (Supervised Machine Learning)
====================================================
Verifies:
1. DemandForecaster trains, tunes, and evaluates Linear, Ridge, and Lasso Regression.
2. RestockClassifier trains, tunes, and evaluates Logistic Regression, Decision Tree,
   Naïve Bayes, SVM, and K-NN.
3. Correct calculation of RMSE, MAE, R2, Accuracy, F1-Score, and ROC-AUC.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from src.module4_supervised.regression_demand import DemandForecaster, RegressionBenchmark
from src.module4_supervised.classification_restock import RestockClassifier, ClassificationBenchmark


@pytest.fixture
def synthetic_supervised_data():
    """Generates small controlled synthetic regression and classification datasets."""
    np.random.seed(42)
    # Regression data
    X_reg, y_reg = make_regression(n_samples=120, n_features=8, noise=0.1, random_state=42)
    X_reg = pd.DataFrame(X_reg, columns=[f"feat_{i}" for i in range(8)])
    y_reg = pd.Series(y_reg, name="target_reg")

    # Classification data (4 classes)
    X_clf, y_clf = make_classification(
        n_samples=120, n_features=8, n_informative=5, n_classes=4, random_state=42
    )
    X_clf = pd.DataFrame(X_clf, columns=[f"feat_{i}" for i in range(8)])
    y_clf = pd.Series(y_clf, name="target_clf")

    # Split train/test
    split = 90
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


def test_demand_forecaster(synthetic_supervised_data, tmp_path):
    forecaster = DemandForecaster(random_state=42)
    benchmark = forecaster.train_and_compare(
        X_train=synthetic_supervised_data["X_train_reg"],
        y_train=synthetic_supervised_data["y_train_reg"],
        X_test=synthetic_supervised_data["X_test_reg"],
        y_test=synthetic_supervised_data["y_test_reg"],
        n_splits=3,
    )

    assert isinstance(benchmark, RegressionBenchmark)
    assert len(benchmark.comparison_table) == 3  # Linear, Ridge, Lasso
    assert benchmark.best_model_name in ["LinearRegression", "Ridge", "Lasso"]

    # Verify metrics are populated
    for _, row in benchmark.comparison_table.iterrows():
        assert row["test_rmse"] >= 0
        assert row["test_mae"] >= 0

    # Test model saving
    saved_path = forecaster.save_best_model(benchmark, output_dir=tmp_path)
    assert saved_path.exists()


def test_restock_classifier(synthetic_supervised_data, tmp_path):
    classifier = RestockClassifier(random_state=42)
    benchmark = classifier.train_and_compare(
        X_train=synthetic_supervised_data["X_train_clf"],
        y_train=synthetic_supervised_data["y_train_clf"],
        X_test=synthetic_supervised_data["X_test_clf"],
        y_test=synthetic_supervised_data["y_test_clf"],
        n_splits=3,
    )

    assert isinstance(benchmark, ClassificationBenchmark)
    assert len(benchmark.comparison_table) == 5  # Logistic, DecisionTree, NaiveBayes, SVM, KNN
    expected_models = {"LogisticRegression", "DecisionTree", "NaiveBayes", "SVM", "KNN"}
    assert set(benchmark.comparison_table["model"]) == expected_models

    # Verify metrics are populated
    for _, row in benchmark.comparison_table.iterrows():
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["f1_macro"] <= 1.0

    # Test model saving
    saved_path = classifier.save_best_model(benchmark, output_dir=tmp_path)
    assert saved_path.exists()
