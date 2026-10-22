"""
Unit Tests for Module 5 (Unsupervised Machine Learning & Apriori)
================================================================
Verifies:
1. ProductClusterEngine runs K-Means, DBSCAN, and Agglomerative Clustering.
2. Automated Deadstock / Spoilage detection outputs dynamic markdowns.
3. MarketBasketAnalyzer runs Apriori algorithm and mines association rules.
"""

import pytest
import pandas as pd
import numpy as np

from src.module5_unsupervised.clustering import ProductClusterEngine, ClusteringResults
from src.module5_unsupervised.association_rules import MarketBasketAnalyzer, AssociationMiningResults
from src.module1_ingestion.pipeline import run_module1_pipeline
from src.generator.synthetic_data_generator import GroceryDataGenerator, GeneratorConfig


@pytest.fixture(scope="module")
def module5_test_data(tmp_path_factory):
    """Generates a small test dataset and processes it through Module 1."""
    raw_dir = tmp_path_factory.mktemp("raw_m5")
    proc_dir = tmp_path_factory.mktemp("proc_m5")

    config = GeneratorConfig(
        num_products=12,
        start_date="2024-01-01",
        num_days=30,
        avg_transactions_per_day=30,
        random_seed=88,
        output_dir=raw_dir,
        inject_anomalies=False,
    )
    generator = GroceryDataGenerator(config)
    products, inventory, sales = generator.generate_all()
    generator.save_to_csv(raw_dir)

    features_df, clean_sales, _ = run_module1_pipeline(raw_dir=raw_dir, processed_dir=proc_dir)
    return {
        "features_df": features_df,
        "clean_sales": clean_sales,
        "products_df": products,
    }


def test_product_cluster_engine(module5_test_data):
    engine = ProductClusterEngine(random_state=42)
    results = engine.fit_and_segment(module5_test_data["features_df"])

    assert isinstance(results, ClusteringResults)
    assert len(results.sku_profiles) == 12  # 12 SKUs

    # Verify cluster columns exist
    expected_cols = [
        "kmeans_cluster",
        "product_health_tier",
        "dbscan_cluster",
        "hierarchical_cluster",
        "days_of_inventory_remaining",
        "turn_ratio",
    ]
    for col in expected_cols:
        assert col in results.sku_profiles.columns, f"Missing cluster profile column: {col}"

    assert results.optimal_k in [3, 4, 5]
    assert results.kmeans_silhouette > 0.1
    assert results.hierarchical_silhouette > 0.1

    # Verify Deadstock & Spoilage Report
    assert isinstance(results.deadstock_report, pd.DataFrame)
    if len(results.deadstock_report) > 0:
        assert (results.deadstock_report["recommended_markdown_pct"] >= 0.10).all()
        assert (results.deadstock_report["recommended_markdown_pct"] <= 0.50).all()
        assert "recommended_action" in results.deadstock_report.columns


def test_market_basket_apriori_analyzer(module5_test_data):
    analyzer = MarketBasketAnalyzer(min_support=0.01, min_lift=1.1, min_confidence=0.15)
    results = analyzer.mine_rules(
        sales_df=module5_test_data["clean_sales"],
        products_df=module5_test_data["products_df"],
    )

    assert isinstance(results, AssociationMiningResults)
    assert len(results.frequent_itemsets) > 0
    assert "support" in results.frequent_itemsets.columns

    if len(results.all_rules) > 0:
        expected_rule_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        for col in expected_rule_cols:
            assert col in results.all_rules.columns
        assert (results.all_rules["lift"] >= 1.0).all()
