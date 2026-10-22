"""
Unit Tests for Module 2 (ML Basics & Model Engineering) and Module 3 (Feature Selection)
========================================================================================
Verifies:
1. MLDataPreprocessor cleans, scales, encodes, and partitions data without leakage.
2. ModelTuner performs K-Fold CV, evaluates Overfitting/Underfitting, and tunes hyperparameters.
3. FeatureSelector executes Filter, Wrapper, and Embedded methods and produces consensus rankings.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from src.module2_engineering.preprocessing import MLDataPreprocessor, SplitStrategy
from src.module2_engineering.validation_tuning import ModelTuner
from src.module3_feature_selection.feature_selection import FeatureSelector
from src.module1_ingestion.pipeline import run_module1_pipeline
from src.generator.synthetic_data_generator import GroceryDataGenerator, GeneratorConfig


@pytest.fixture(scope="module")
def prepared_features(tmp_path_factory):
    """Generates a small test dataset and runs Module 1 to produce feature matrix."""
    raw_dir = tmp_path_factory.mktemp("raw_m2_m3")
    proc_dir = tmp_path_factory.mktemp("proc_m2_m3")

    config = GeneratorConfig(
        num_products=10,
        start_date="2024-01-01",
        num_days=30,
        avg_transactions_per_day=20,
        random_seed=77,
        output_dir=raw_dir,
        inject_anomalies=False,
    )
    generator = GroceryDataGenerator(config)
    generator.save_to_csv(raw_dir)

    features_df, _, _ = run_module1_pipeline(raw_dir=raw_dir, processed_dir=proc_dir)
    return features_df


def test_ml_data_preprocessor(prepared_features):
    preprocessor = MLDataPreprocessor(scaler_type="robust", test_ratio=0.20, n_splits=3)
    dataset = preprocessor.prepare_dataset(prepared_features, split_strategy=SplitStrategy.TEMPORAL)

    # Check shapes
    assert len(dataset.X_train) > 0
    assert len(dataset.X_test) > 0
    assert len(dataset.X_train) + len(dataset.X_test) == len(prepared_features)
    assert len(dataset.y_train_reg) == len(dataset.X_train)
    assert len(dataset.y_test_reg) == len(dataset.X_test)
    assert len(dataset.y_train_clf) == len(dataset.X_train)

    # Check no NaNs in transformed feature spaces
    assert not dataset.X_train.isna().any().any()
    assert not dataset.X_test.isna().any().any()

    # Check CV splitters
    splitters = preprocessor.get_cv_splitters()
    assert "standard_kfold" in splitters
    assert "stratified_kfold" in splitters
    assert "time_series_split" in splitters


def test_model_tuner_kfold_and_tuning(prepared_features):
    preprocessor = MLDataPreprocessor(scaler_type="standard", test_ratio=0.20, n_splits=3)
    dataset = preprocessor.prepare_dataset(prepared_features, split_strategy=SplitStrategy.TEMPORAL)

    tuner = ModelTuner(random_state=42)
    splitters = preprocessor.get_cv_splitters()

    # 1. Test K-Fold validation
    model = Ridge(alpha=1.0)
    val_res = tuner.validate_with_kfold(
        model=model,
        model_name="RidgeBaseline",
        X=dataset.X_train,
        y=dataset.y_train_reg,
        cv=splitters["standard_kfold"],
        scoring="r2",
    )

    assert val_res.model_name == "RidgeBaseline"
    assert isinstance(val_res.train_scores_mean, float)
    assert isinstance(val_res.val_scores_mean, float)
    assert isinstance(val_res.overfitting_gap, float)

    # 2. Test Hyperparameter Tuning
    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    tuning_res = tuner.tune_hyperparameters(
        model=Ridge(),
        model_name="RidgeTuned",
        param_grid=param_grid,
        X=dataset.X_train,
        y=dataset.y_train_reg,
        cv=splitters["standard_kfold"],
        search_type="grid",
        scoring="r2",
    )

    assert "alpha" in tuning_res.best_params
    assert tuning_res.best_estimator is not None
    assert len(tuning_res.cv_results_df) == 3


def test_feature_selector_multi_methods(prepared_features):
    preprocessor = MLDataPreprocessor(scaler_type="robust", test_ratio=0.20, n_splits=3)
    dataset = preprocessor.prepare_dataset(prepared_features, split_strategy=SplitStrategy.TEMPORAL)

    selector = FeatureSelector(random_state=42)
    summary = selector.select_features(
        X_train=dataset.X_train,
        y_train=dataset.y_train_reg,
        top_k=8,
    )

    assert len(summary.selected_features) == 8
    assert len(summary.filter_top_k) == 8
    assert len(summary.rf_top_k) == 8
    assert len(summary.rfe_selected) == 8
    assert len(summary.ranking_table) == len(dataset.X_train.columns)

    # Check ranking table structure
    expected_cols = ["feature", "f_score", "kbest_rank", "rf_importance", "rf_rank", "rfe_rank", "lasso_coef", "lasso_rank", "composite_rank_score"]
    for c in expected_cols:
        assert c in summary.ranking_table.columns
