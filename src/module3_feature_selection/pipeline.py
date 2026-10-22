"""
Module 3 Feature Selection Pipeline
===================================
Applies Filter, Wrapper, and Embedded methods to the processed dataset.
Persists the consensus rankings and selected feature subsets.
"""

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.module2_engineering.preprocessing import MLDataPreprocessor, SplitStrategy
from src.module3_feature_selection.feature_selection import FeatureSelectionSummary, FeatureSelector


def run_module3_pipeline(
    features_path: Path = PROCESSED_DATA_DIR / "daily_product_features.csv",
    reports_dir: Path = REPORTS_DIR,
    top_k: int = 15,
) -> Tuple[FeatureSelectionSummary, Path]:
    """
    Executes Feature Selection across Filter, Wrapper, and Embedded algorithms.
    """
    print("\n=====================================================================")
    print("      MODULE 3: MULTI-METHOD FEATURE SELECTION PIPELINE              ")
    print("=====================================================================")

    df = pd.read_csv(features_path)
    preprocessor = MLDataPreprocessor(scaler_type="robust", test_ratio=0.20)
    dataset = preprocessor.prepare_dataset(df, split_strategy=SplitStrategy.TEMPORAL)

    selector = FeatureSelector(random_state=42)
    summary = selector.select_features(
        X_train=dataset.X_train,
        y_train=dataset.y_train_reg,
        top_k=top_k,
    )

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_csv = reports_dir / "feature_selection_rankings.csv"
    summary.ranking_table.to_csv(report_csv, index=False)

    json_path = PROCESSED_DATA_DIR / "selected_features.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "consensus_features": summary.selected_features,
                "filter_kbest_features": summary.filter_top_k,
                "rf_importance_features": summary.rf_top_k,
                "rfe_features": summary.rfe_selected,
                "lasso_active_features": summary.lasso_active_features,
            },
            f,
            indent=2,
        )

    print(f"\n[Module 3 - Output]")
    print(f"  - Ranking Table:    {report_csv}")
    print(f"  - Selected Features: {json_path}")
    print("=====================================================================")
    print("      MODULE 3 COMPLETED SUCCESSFULLY                               ")
    print("=====================================================================\n")

    return summary, report_csv


if __name__ == "__main__":
    run_module3_pipeline()
