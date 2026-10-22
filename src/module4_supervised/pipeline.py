"""
Module 4 (UNIT-III) Orchestration Pipeline
==========================================
Runs supervised regression and classification benchmarking pipelines
using consensus-selected features from Module 3.
"""

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.module2_engineering.preprocessing import MLDataPreprocessor, SplitStrategy
from src.module4_supervised.classification_restock import ClassificationBenchmark, RestockClassifier
from src.module4_supervised.regression_demand import DemandForecaster, RegressionBenchmark


def run_module4_pipeline(
    features_path: Path = PROCESSED_DATA_DIR / "daily_product_features.csv",
    selected_features_path: Path = PROCESSED_DATA_DIR / "selected_features.json",
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> Tuple[RegressionBenchmark, ClassificationBenchmark]:
    """
    Executes Module 4 supervised pipelines:
    1. Demand regression benchmark (Linear, Ridge, Lasso)
    2. Restock urgency classification benchmark (Logistic, Decision Tree, Naïve Bayes, SVM, KNN)
    """
    print("\n=====================================================================")
    print("      MODULE 4: SUPERVISED MACHINE LEARNING BENCHMARK PIPELINE       ")
    print("=====================================================================")

    df = pd.read_csv(features_path)
    preprocessor = MLDataPreprocessor(scaler_type="robust", test_ratio=0.20)
    dataset = preprocessor.prepare_dataset(df, split_strategy=SplitStrategy.TEMPORAL)

    # Load selected consensus features if available
    if Path(selected_features_path).exists():
        with open(selected_features_path, "r") as f:
            selected_info = json.load(f)
            consensus_features = [f for f in selected_info["consensus_features"] if f in dataset.X_train.columns]
        print(f"[Module 4] Using top {len(consensus_features)} consensus features from Module 3.")
        X_train_sub = dataset.X_train[consensus_features]
        X_test_sub = dataset.X_test[consensus_features]
    else:
        print("[Module 4] Consensus features not found. Using full preprocessed feature space.")
        X_train_sub = dataset.X_train
        X_test_sub = dataset.X_test

    # 1. Demand & Sales Forecasting Regression
    forecaster = DemandForecaster(random_state=42)
    reg_benchmark = forecaster.train_and_compare(
        X_train=X_train_sub,
        y_train=dataset.y_train_reg,
        X_test=X_test_sub,
        y_test=dataset.y_test_reg,
        n_splits=5,
    )
    forecaster.save_best_model(reg_benchmark, output_dir=models_dir)

    # 2. Restock Urgency Action Classification
    classifier = RestockClassifier(random_state=42)
    clf_benchmark = classifier.train_and_compare(
        X_train=X_train_sub,
        y_train=dataset.y_train_clf,
        X_test=X_test_sub,
        y_test=dataset.y_test_clf,
        n_splits=5,
    )
    classifier.save_best_model(clf_benchmark, output_dir=models_dir)

    # Persist benchmark tables
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    reg_benchmark.comparison_table.to_csv(reports_dir / "regression_comparison.csv", index=False)
    clf_benchmark.comparison_table.to_csv(reports_dir / "classification_comparison.csv", index=False)

    print("\n=====================================================================")
    print("      MODULE 4 COMPLETED SUCCESSFULLY                               ")
    print("=====================================================================\n")

    return reg_benchmark, clf_benchmark


if __name__ == "__main__":
    run_module4_pipeline()
