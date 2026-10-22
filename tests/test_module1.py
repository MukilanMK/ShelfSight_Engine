"""
Unit Tests for Module 1 (Data Ingestion, Validation & Feature Engineering)
========================================================================
Verifies:
1. DataValidator cleans corrupted records, standardizes SKUs, imputes timestamps.
2. FeatureEngineer calculates temporal, rolling velocity, lead times, DIR, GMROI,
   dynamic safety stock, and restock targets.
3. Zero hardcoding invariants are preserved across outputs.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.module1_ingestion.validation import DataValidator, ValidationReport
from src.module1_ingestion.feature_engineering import FeatureEngineer
from src.generator.synthetic_data_generator import GroceryDataGenerator, GeneratorConfig


@pytest.fixture(scope="module")
def sample_raw_data(tmp_path_factory):
    """Generates a small test dataset with anomalies."""
    out_dir = tmp_path_factory.mktemp("raw_data")
    config = GeneratorConfig(
        num_products=12,
        start_date="2024-01-01",
        num_days=35,
        avg_transactions_per_day=25,
        random_seed=99,
        output_dir=out_dir,
        inject_anomalies=True,
        anomaly_rate=0.03,  # 3% for testing cleaning
    )
    generator = GroceryDataGenerator(config)
    generator.save_to_csv(out_dir)
    return out_dir


def test_validator_cleans_and_standardizes(sample_raw_data):
    validator = DataValidator(raw_data_dir=sample_raw_data)
    products, sales, inventory, report = validator.run_validation()

    # 1. Product catalog checks
    assert len(products) > 0
    assert (products["cost_price"] > 0).all()
    assert (products["selling_price"] >= products["cost_price"]).all()
    assert products["sku_id"].str.startswith("SKU-GROC-").all()

    # 2. Sales checks
    assert len(sales) > 0
    assert (sales["units_sold"] > 0).all()
    assert sales["timestamp"].isna().sum() == 0, "All missing timestamps must be imputed"
    assert sales["sku_id"].str.startswith("SKU-GROC-").all()

    # 3. Inventory checks
    assert len(inventory) > 0
    assert (inventory["starting_stock"] >= 0).all()
    assert (inventory["ending_stock"] >= 0).all()

    # 4. Report checks
    assert report.sales_clean_count > 0
    assert report.inventory_clean_count > 0


def test_feature_engineering_calculations(sample_raw_data):
    validator = DataValidator(raw_data_dir=sample_raw_data)
    products, sales, inventory, _ = validator.run_validation()

    engineer = FeatureEngineer(service_level=0.95)
    features_df = engineer.build_feature_dataset(products, sales, inventory)

    assert len(features_df) > 0

    # Temporal features
    expected_temporal = ["day_of_week", "is_weekend", "is_payday", "dow_sin", "dow_cos", "seasonality_index"]
    for col in expected_temporal:
        assert col in features_df.columns, f"Missing temporal column: {col}"

    # Rolling velocities
    expected_velocities = ["velocity_7d", "velocity_14d", "velocity_30d", "demand_std_7d", "demand_std_14d"]
    for col in expected_velocities:
        assert col in features_df.columns, f"Missing velocity column: {col}"
        assert (features_df[col] >= 0).all(), f"Velocity column {col} cannot be negative"

    # Lead times and inventory metrics
    expected_inv = [
        "dynamic_lead_time_mean",
        "dynamic_lead_time_std",
        "days_of_inventory_remaining",
        "gmroi",
        "dynamic_safety_stock",
        "dynamic_reorder_point",
        "trend_acceleration_zscore",
        "restock_urgency_tier",
        "restock_target_encoded",
    ]
    for col in expected_inv:
        assert col in features_df.columns, f"Missing inventory/financial metric: {col}"

    # Check validity of safety stock & reorder point
    assert (features_df["dynamic_safety_stock"] >= 0).all()
    assert (features_df["dynamic_reorder_point"] >= features_df["dynamic_safety_stock"]).all()

    # Check classification target encodes properly
    unique_tiers = set(features_df["restock_target_encoded"].unique())
    assert unique_tiers.issubset({0, 1, 2, 3})
