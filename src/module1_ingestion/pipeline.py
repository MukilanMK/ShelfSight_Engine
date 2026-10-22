"""
Module 1 Orchestration Pipeline
===============================
Runs ingestion, validation, and feature engineering, then persists
curated datasets into `data/processed/` for downstream modeling modules.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.module1_ingestion.feature_engineering import FeatureEngineer
from src.module1_ingestion.validation import DataValidator, ValidationReport


def run_module1_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    """
    Executes Module 1:
    1. Validates and cleans raw data (products, sales, inventory).
    2. Builds daily SKU-level feature matrix with temporal, velocity, financial, and inventory metrics.
    3. Exports cleaned market-basket transactions and master feature matrix.
    """
    print("\n=====================================================================")
    print("      MODULE 1: DATA INGESTION, VALIDATION & FEATURE ENGINEERING     ")
    print("=====================================================================")

    # 1. Validation & Cleaning
    validator = DataValidator(raw_data_dir=raw_dir)
    products_clean, sales_clean, inventory_clean, report = validator.run_validation()

    print("\n[Module 1 - Ingestion Audit]")
    print(f"  - Products:  {report.products_raw_count} raw -> {report.products_clean_count} clean")
    print(f"  - Sales:     {report.sales_raw_count} raw -> {report.sales_clean_count} clean")
    print(f"               (Standardized SKUs: {report.sales_skus_standardized}, Imputed TS: {report.sales_timestamps_imputed}, Dropped invalid: {report.sales_invalid_records_dropped})")
    print(f"  - Inventory: {report.inventory_raw_count} raw -> {report.inventory_clean_count} clean")
    print(f"               (Dropped invalid: {report.inventory_invalid_dropped})")

    # 2. Feature Engineering
    engineer = FeatureEngineer(service_level=0.95)
    master_features_df = engineer.build_feature_dataset(
        products_df=products_clean,
        sales_df=sales_clean,
        inventory_df=inventory_clean,
    )

    # 3. Persist Processed Artifacts
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    features_out = processed_dir / "daily_product_features.csv"
    transactions_out = processed_dir / "cleaned_transactions.csv"

    print(f"\n[Module 1 - Output]")
    print(f"  - Writing daily master feature matrix: {features_out} ({len(master_features_df)} records)")
    master_features_df.to_csv(features_out, index=False)

    print(f"  - Writing cleaned market transactions: {transactions_out} ({len(sales_clean)} records)")
    sales_clean.to_csv(transactions_out, index=False)

    print("=====================================================================")
    print("      MODULE 1 COMPLETED SUCCESSFULLY                               ")
    print("=====================================================================\n")

    return master_features_df, sales_clean, report


if __name__ == "__main__":
    run_module1_pipeline()
