"""
Unit Tests for Synthetic Data Generator
======================================
Verifies:
1. Schema conformity across all 3 generated datasets.
2. Referential integrity between products, inventory, and sales.
3. Market basket structure for Association Rule Mining.
4. Non-empty data generation and statistical properties.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.generator.synthetic_data_generator import GroceryDataGenerator, GeneratorConfig


@pytest.fixture
def clean_generator():
    """Generator instance with anomalies disabled for strict schema checks."""
    config = GeneratorConfig(
        num_products=15,
        start_date="2024-01-01",
        num_days=10,
        avg_transactions_per_day=20,
        random_seed=123,
        inject_anomalies=False,
    )
    return GroceryDataGenerator(config)


def test_product_master_schema(clean_generator):
    products_df, _, _ = clean_generator.generate_all()

    required_cols = [
        "sku_id", "upc", "product_name", "category",
        "cost_price", "selling_price", "margin_pct",
        "is_perishable", "shelf_life_days", "storage_type",
        "supplier_lead_time_mean", "supplier_lead_time_std", "baseline_tier"
    ]
    for col in required_cols:
        assert col in products_df.columns, f"Missing product column: {col}"

    assert len(products_df) == 15
    assert (products_df["selling_price"] >= products_df["cost_price"]).all()
    assert (products_df["shelf_life_days"] > 0).all()
    assert products_df["sku_id"].is_unique


def test_sales_transactions_schema(clean_generator):
    products_df, _, sales_df = clean_generator.generate_all()

    required_cols = [
        "transaction_id", "basket_id", "date", "timestamp",
        "customer_id", "sku_id", "units_sold", "unit_price",
        "discount_pct", "total_amount"
    ]
    for col in required_cols:
        assert col in sales_df.columns, f"Missing sales column: {col}"

    assert len(sales_df) > 0
    assert (sales_df["units_sold"] > 0).all()
    assert (sales_df["unit_price"] > 0).all()
    assert (sales_df["total_amount"] >= 0).all()

    # Verify foreign key to products
    product_skus = set(products_df["sku_id"])
    sales_skus = set(sales_df["sku_id"])
    assert sales_skus.issubset(product_skus)

    # Verify multi-item basket presence for Association Rules
    basket_sizes = sales_df.groupby("basket_id").size()
    assert (basket_sizes > 1).any(), "Expected some multi-item baskets for Apriori mining"


def test_inventory_snapshots_schema(clean_generator):
    products_df, inventory_df, _ = clean_generator.generate_all()

    required_cols = [
        "date", "sku_id", "starting_stock", "restock_received",
        "sales_units", "unfulfilled_demand", "spoilage_units",
        "ending_stock", "batch_expiry_date", "supplier_lead_time_recorded"
    ]
    for col in required_cols:
        assert col in inventory_df.columns, f"Missing inventory column: {col}"

    assert len(inventory_df) == 15 * 10  # 15 SKUs across 10 days
    assert (inventory_df["ending_stock"] >= 0).all()
    assert (inventory_df["starting_stock"] >= 0).all()


def test_save_to_csv(tmp_path):
    config = GeneratorConfig(
        num_products=10,
        start_date="2024-01-01",
        num_days=5,
        avg_transactions_per_day=10,
        random_seed=42,
        output_dir=tmp_path,
        inject_anomalies=False,
    )
    generator = GroceryDataGenerator(config)
    paths = generator.save_to_csv(tmp_path)

    assert paths["products"].exists()
    assert paths["inventory"].exists()
    assert paths["sales"].exists()

    df_p = pd.read_csv(paths["products"])
    assert len(df_p) == 10
