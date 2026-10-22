"""
Module 6 (UNIT-V) Orchestration Pipeline
=========================================
Runs advanced ensemble benchmarks:
- Bagging (Random Forest)
- Boosting (AdaBoost, Gradient Boosting, XGBoost)
Persists performance leaderboards, justification reports, and final models.
"""

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.module2_engineering.preprocessing import MLDataPreprocessor, SplitStrategy
from src.module6_ensemble.ensemble_models import EnsembleBenchmark, EnsembleModelEngine


def run_module6_pipeline(
    features_path: Path = PROCESSED_DATA_DIR / "daily_product_features.csv",
    selected_features_path: Path = PROCESSED_DATA_DIR / "selected_features.json",
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> EnsembleBenchmark:
    """
    Executes Module 6 ensemble benchmarking.
    """
    print("\n=====================================================================")
    print("      MODULE 6: ADVANCED ENSEMBLE LEARNING & BOOSTING PIPELINE       ")
    print("=====================================================================")

    df = pd.read_csv(features_path)
    preprocessor = MLDataPreprocessor(scaler_type="robust", test_ratio=0.20)
    dataset = preprocessor.prepare_dataset(df, split_strategy=SplitStrategy.TEMPORAL)

    # Use consensus features from Module 3
    if Path(selected_features_path).exists():
        with open(selected_features_path, "r") as f:
            selected_info = json.load(f)
            consensus_features = [f for f in selected_info["consensus_features"] if f in dataset.X_train.columns]
        print(f"[Module 6] Benchmarking ensembles using {len(consensus_features)} consensus features.")
        X_train_sub = dataset.X_train[consensus_features]
        X_test_sub = dataset.X_test[consensus_features]
    else:
        X_train_sub = dataset.X_train
        X_test_sub = dataset.X_test

    engine = EnsembleModelEngine(random_state=42)

    # 1. Regression Ensembles
    print("\n--- Benchmarking Regression Ensembles (Bagging & Boosting) ---")
    reg_df, best_reg_name, best_reg_model = engine.benchmark_regression_ensembles(
        X_train=X_train_sub,
        y_train=dataset.y_train_reg,
        X_test=X_test_sub,
        y_test=dataset.y_test_reg,
        n_splits=5,
    )

    # 2. Classification Ensembles
    print("\n--- Benchmarking Classification Ensembles (Bagging & Boosting) ---")
    clf_df, best_clf_name, best_clf_model = engine.benchmark_classification_ensembles(
        X_train=X_train_sub,
        y_train=dataset.y_train_clf,
        X_test=X_test_sub,
        y_test=dataset.y_test_clf,
        n_splits=5,
    )

    # 3. Model Justification Report
    justification_md = engine.generate_justification_report(
        reg_df=reg_df,
        clf_df=clf_df,
        best_reg_name=best_reg_name,
        best_clf_name=best_clf_name,
    )

    # 4. Persist Artifacts
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    reg_out = reports_dir / "ensemble_regression_comparison.csv"
    clf_out = reports_dir / "ensemble_classification_comparison.csv"
    just_out = reports_dir / "model_justification_report.md"

    reg_df.to_csv(reg_out, index=False)
    clf_df.to_csv(clf_out, index=False)
    with open(just_out, "w") as f:
        f.write(justification_md)

    joblib.dump(best_reg_model, models_dir / "final_ensemble_demand_forecaster.pkl")
    joblib.dump(best_clf_model, models_dir / "final_ensemble_restock_classifier.pkl")

    print(f"\n[Module 6 - Output Reports & Models]")
    print(f"  - Regression Leaderboard:    {reg_out}")
    print(f"  - Classification Leaderboard: {clf_out}")
    print(f"  - Justification Report:       {just_out}")
    print(f"  - Best Forecaster Model:      {models_dir / 'final_ensemble_demand_forecaster.pkl'} ({best_reg_name})")
    print(f"  - Best Restock Model:         {models_dir / 'final_ensemble_restock_classifier.pkl'} ({best_clf_name})")
    print("=====================================================================")
    print("      MODULE 6 COMPLETED SUCCESSFULLY                               ")
    print("=====================================================================\n")

    return EnsembleBenchmark(
        regression_leaderboard=reg_df,
        classification_leaderboard=clf_df,
        best_regressor_name=best_reg_name,
        best_regressor=best_reg_model,
        best_classifier_name=best_clf_name,
        best_classifier=best_clf_model,
        justification_report=justification_md,
    )


if __name__ == "__main__":
    run_module6_pipeline()
