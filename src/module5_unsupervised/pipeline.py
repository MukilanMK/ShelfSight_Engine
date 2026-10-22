"""
Module 5 (UNIT-IV) Orchestration Pipeline
=========================================
Runs unsupervised learning:
1. Product health clustering (K-Means, DBSCAN, Hierarchical).
2. Automated deadstock and spoilage detection with dynamic markdowns.
3. Apriori association rule mining for slow-mover basket bundling.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
from src.module5_unsupervised.association_rules import AssociationMiningResults, MarketBasketAnalyzer
from src.module5_unsupervised.clustering import ClusteringResults, ProductClusterEngine


def run_module5_pipeline(
    features_path: Path = PROCESSED_DATA_DIR / "daily_product_features.csv",
    products_path: Path = RAW_DATA_DIR / "products.csv",
    sales_path: Path = PROCESSED_DATA_DIR / "cleaned_transactions.csv",
    reports_dir: Path = REPORTS_DIR,
) -> Tuple[ClusteringResults, AssociationMiningResults]:
    """
    Executes Module 5 unsupervised learning pipelines.
    """
    print("\n=====================================================================")
    print("      MODULE 5: UNSUPERVISED MACHINE LEARNING & DEADSTOCK PIPELINE   ")
    print("=====================================================================")

    features_df = pd.read_csv(features_path)
    products_df = pd.read_csv(products_path)
    sales_df = pd.read_csv(sales_path)

    # 1. Product Clustering & Deadstock Detection
    engine = ProductClusterEngine(random_state=42)
    clustering_results = engine.fit_and_segment(features_df)

    # Extract identified at-risk SKU IDs
    at_risk_skus = set(clustering_results.deadstock_report["sku_id"].unique())

    # 2. Market Basket Association Rules (Apriori)
    analyzer = MarketBasketAnalyzer(min_support=0.012, min_lift=1.20, min_confidence=0.20)
    association_results = analyzer.mine_rules(
        sales_df=sales_df,
        products_df=products_df,
        deadstock_skus=at_risk_skus,
    )

    # 3. Export Reports
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    clusters_out = reports_dir / "product_clusters.csv"
    clustering_results.sku_profiles.to_csv(clusters_out, index=False)

    deadstock_out = reports_dir / "deadstock_spoilage_risk_report.csv"
    clustering_results.deadstock_report.to_csv(deadstock_out, index=False)

    rules_out = reports_dir / "association_rules_bundles.csv"
    association_results.slow_mover_bundles.to_csv(rules_out, index=False)

    print(f"\n[Module 5 - Output Reports]")
    print(f"  - Product Clusters:          {clusters_out} ({len(clustering_results.sku_profiles)} SKUs)")
    print(f"  - Deadstock/Spoilage Report: {deadstock_out} ({len(clustering_results.deadstock_report)} flagged items)")
    print(f"  - Market Basket Bundles:     {rules_out} ({len(association_results.slow_mover_bundles)} bundle pairs)")
    print("=====================================================================")
    print("      MODULE 5 COMPLETED SUCCESSFULLY                               ")
    print("=====================================================================\n")

    return clustering_results, association_results


if __name__ == "__main__":
    run_module5_pipeline()
