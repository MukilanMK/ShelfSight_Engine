"""
End-to-End Retail Machine Learning System Pipeline Runner
=========================================================
Runs all syllabus modules sequentially:
1. Ingestion, Validation & Zero-Hardcoded Feature Engineering (Module 1)
2. ML Preprocessing, K-Fold Validation & Hyperparameter Tuning (Module 2)
3. Multi-Method Feature Selection: Filter, Wrapper, Embedded (Module 3)
4. Supervised Machine Learning: Demand Regression & Restock Classification (Module 4)
5. Unsupervised Machine Learning: K-Means, DBSCAN, Hierarchy, & Apriori (Module 5)
6. Advanced Ensemble Learning: Random Forest, AdaBoost, GBDT, & XGBoost (Module 6)
7. Executive Summary Dashboard Generator (Deliverable 3)

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-train (to regenerate dashboard from existing models)
"""

import argparse
import sys
from pathlib import Path

from src.dashboard.summary_reporter import ExecutiveSummaryReporter
from src.module1_ingestion.pipeline import run_module1_pipeline
from src.module3_feature_selection.pipeline import run_module3_pipeline
from src.module4_supervised.pipeline import run_module4_pipeline
from src.module5_unsupervised.pipeline import run_module5_pipeline
from src.module6_ensemble.pipeline import run_module6_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Grocery Store ML Pipeline Master Runner")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training and run dashboard directly")
    args = parser.parse_args()

    print("\n" + "=" * 90)
    print("    STARTING END-TO-END RETAIL SUPPLY CHAIN MACHINE LEARNING PIPELINE      ")
    print("=" * 90)

    if not args.skip_train:
        # Step 1: Module 1
        print("\n>>> STEP 1: Running Module 1 (Ingestion & Feature Engineering)...")
        run_module1_pipeline()

        # Step 2: Module 3
        print("\n>>> STEP 2: Running Module 3 (Multi-Method Feature Selection)...")
        run_module3_pipeline()

        # Step 3: Module 4
        print("\n>>> STEP 3: Running Module 4 (Supervised Learning Benchmarks)...")
        run_module4_pipeline()

        # Step 4: Module 5
        print("\n>>> STEP 4: Running Module 5 (Unsupervised Segmentation & Deadstock)...")
        run_module5_pipeline()

        # Step 5: Module 6
        print("\n>>> STEP 5: Running Module 6 (Advanced Ensemble Boosting)...")
        run_module6_pipeline()

    # Step 6: Executive Dashboard Deliverable
    print("\n>>> STEP 6: Compiling Executive Summary Dashboard...")
    reporter = ExecutiveSummaryReporter()
    reporter.generate_dashboard()

    print("\n" + "=" * 90)
    print("    PIPELINE EXECUTION COMPLETE! ALL DELIVERABLES GENERATED SUCCESSFULLY    ")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
