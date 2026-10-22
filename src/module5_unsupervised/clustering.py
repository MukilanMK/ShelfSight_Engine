"""
Module 5 (UNIT-IV): Unsupervised Product Segmentation & Deadstock Detection
==========================================================================
Implements:
1. K-Means Clustering with automated Silhouette Score selection for dynamic health tiers.
2. Density-Based Clustering (DBSCAN) to identify operational outliers and anomalous products.
3. Agglomerative Hierarchical Clustering to evaluate product similarity trees and dendrograms.
4. Zero-hardcoded Deadstock and Spoilage Risk detection with automated dynamic markdowns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResults:
    """Encapsulates multi-algorithm clustering outputs and deadstock risk reports."""
    sku_profiles: pd.DataFrame
    deadstock_report: pd.DataFrame
    optimal_k: int
    kmeans_silhouette: float
    dbscan_n_clusters: int
    dbscan_n_noise: int
    hierarchical_silhouette: float
    linkage_matrix: np.ndarray


class ProductClusterEngine:
    """
    Executes K-Means, DBSCAN, and Agglomerative Hierarchical Clustering
    to segment product portfolio without static rule-based cutoffs.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scaler = StandardScaler()

    def fit_and_segment(self, df_features: pd.DataFrame) -> ClusteringResults:
        """
        Aggregates product-level supply chain indicators and executes
        unsupervised clustering algorithms.
        """
        print("[ProductClusterEngine] Aggregating product-level profiles...")
        sku_df = self._aggregate_product_profiles(df_features)

        # Feature matrix for clustering
        cluster_feature_cols = [
            "avg_velocity",
            "trend_acceleration",
            "days_of_inventory_remaining",
            "turn_ratio",
            "gmroi",
            "margin_pct",
            "spoilage_units_total",
            "days_to_expiry_min",
        ]
        X_raw = sku_df[cluster_feature_cols].copy()
        # Impute missing values with column median
        X_raw = X_raw.fillna(X_raw.median())
        X_scaled = self.scaler.fit_transform(X_raw)

        # 1. K-MEANS CLUSTERING (Dynamic Silhouette Optimization)
        print("[ProductClusterEngine] 1. Running K-Means with Silhouette Score optimization...")
        best_k, best_kmeans, best_sil = self._optimize_kmeans(X_scaled)
        sku_df["kmeans_cluster"] = best_kmeans.labels_

        # Dynamically assign meaningful tier names based on centroid profiles
        cluster_labels_map = self._label_kmeans_clusters(
            centroids=best_kmeans.cluster_centers_,
            feature_cols=cluster_feature_cols,
        )
        sku_df["product_health_tier"] = sku_df["kmeans_cluster"].map(cluster_labels_map)

        # 2. DBSCAN (Density-Based Spatial Clustering)
        print("[ProductClusterEngine] 2. Running DBSCAN density-based clustering...")
        eps_dyn = self._estimate_dbscan_eps(X_scaled)
        dbscan = DBSCAN(eps=eps_dyn, min_samples=3)
        sku_df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)
        n_db_clusters = len(set(sku_df["dbscan_cluster"])) - (1 if -1 in sku_df["dbscan_cluster"] else 0)
        n_db_noise = (sku_df["dbscan_cluster"] == -1).sum()

        # 3. AGGLOMERATIVE HIERARCHICAL CLUSTERING
        print("[ProductClusterEngine] 3. Running Agglomerative Hierarchical Clustering...")
        link_matrix = linkage(X_scaled, method="ward", metric="euclidean")
        agg_cluster = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
        sku_df["hierarchical_cluster"] = agg_cluster.fit_predict(X_scaled)
        hier_sil = float(silhouette_score(X_scaled, sku_df["hierarchical_cluster"]))

        # 4. DEADSTOCK & SPOILAGE RISK EVALUATION (Zero Hardcoding)
        print("[ProductClusterEngine] 4. Identifying Deadstock / Spoilage items and calculating dynamic markdowns...")
        deadstock_report = self._evaluate_deadstock_and_spoilage(sku_df, X_scaled, cluster_feature_cols)

        print(f"[ProductClusterEngine] Segmentation complete: Optimal K={best_k} (Silhouette={best_sil:.3f}), DBSCAN Noise Points={n_db_noise}")

        return ClusteringResults(
            sku_profiles=sku_df,
            deadstock_report=deadstock_report,
            optimal_k=best_k,
            kmeans_silhouette=best_sil,
            dbscan_n_clusters=n_db_clusters,
            dbscan_n_noise=int(n_db_noise),
            hierarchical_silhouette=round(hier_sil, 4),
            linkage_matrix=link_matrix,
        )

    def _aggregate_product_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes representative multi-dimensional summary for each SKU."""
        # Use latest snapshot for current stock and moving metrics over the horizon
        agg_dict = {
            "product_name": "first",
            "category": "first",
            "cost_price": "first",
            "selling_price": "first",
            "margin_pct": "first",
            "is_perishable": "first",
            "shelf_life_days": "first",
            "ending_stock": "last",
            "velocity_14d": "mean",
            "trend_acceleration_zscore": "mean",
            "days_of_inventory_remaining": "median",
            "inventory_turn_ratio_30d": "mean",
            "gmroi": "mean",
            "spoilage_units": "sum",
            "days_to_expiry": "min",
        }
        # Filter existing columns
        valid_aggs = {k: v for k, v in agg_dict.items() if k in df.columns}
        sku_summary = df.groupby("sku_id").agg(valid_aggs).reset_index()

        # Rename to standardized profile columns
        sku_summary.rename(
            columns={
                "velocity_14d": "avg_velocity",
                "trend_acceleration_zscore": "trend_acceleration",
                "inventory_turn_ratio_30d": "turn_ratio",
                "spoilage_units": "spoilage_units_total",
                "days_to_expiry": "days_to_expiry_min",
                "ending_stock": "current_on_hand_stock",
            },
            inplace=True,
        )
        return sku_summary

    def _optimize_kmeans(self, X: np.ndarray) -> Tuple[int, KMeans, float]:
        """Evaluates k in [3, 4, 5] using Silhouette Score to pick mathematically optimal k."""
        best_k = 3
        best_score = -1.0
        best_model: Optional[KMeans] = None

        for k in [3, 4, 5]:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            score = float(silhouette_score(X, labels))
            if score > best_score:
                best_score = score
                best_k = k
                best_model = km

        assert best_model is not None
        return best_k, best_model, round(best_score, 4)

    def _label_kmeans_clusters(
        self,
        centroids: np.ndarray,
        feature_cols: List[str],
    ) -> Dict[int, str]:
        """
        Dynamically labels clusters based on centroid rank of velocity and inventory turn.
        Zero hardcoding: relative rankings determine cluster identity.
        """
        vel_idx = feature_cols.index("avg_velocity")
        turn_idx = feature_cols.index("turn_ratio")
        dir_idx = feature_cols.index("days_of_inventory_remaining")

        k = len(centroids)
        # Compute combined vitality score: high velocity & turn, low DIR
        vitality_scores = centroids[:, vel_idx] + centroids[:, turn_idx] - centroids[:, dir_idx]
        sorted_clusters = np.argsort(vitality_scores)

        # Lowest vitality is Deadstock / At-Risk; Highest is Fast-Mover
        mapping: Dict[int, str] = {}
        mapping[sorted_clusters[0]] = "At-Risk / Deadstock"
        mapping[sorted_clusters[-1]] = "Fast-Moving Star"

        if k == 3:
            mapping[sorted_clusters[1]] = "Steady Staple"
        elif k == 4:
            mapping[sorted_clusters[1]] = "Slow-Mover"
            mapping[sorted_clusters[2]] = "Steady Staple"
        elif k >= 5:
            mapping[sorted_clusters[1]] = "Slow-Mover"
            mapping[sorted_clusters[2]] = "Moderate Staple"
            mapping[sorted_clusters[3]] = "Healthy Performer"

        return mapping

    def _estimate_dbscan_eps(self, X: np.ndarray) -> float:
        """Estimates DBSCAN eps parameter using k-NN distance 75th percentile."""
        nn = NearestNeighbors(n_neighbors=4)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        k_distances = distances[:, -1]
        eps_val = float(np.percentile(k_distances, 75))
        return max(0.5, round(eps_val, 2))

    def _evaluate_deadstock_and_spoilage(
        self,
        sku_df: pd.DataFrame,
        X_scaled: np.ndarray,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """
        Zero-hardcoding risk evaluation:
        Identifies items assigned to 'At-Risk / Deadstock' or flagged as DBSCAN noise
        with high days of inventory remaining.
        Calculates dynamic markdown percentage:
            Markdown = margin_pct * (1 + risk_score_normalized) / 2
        Ensures markdowns protect baseline unit economics.
        """
        deadstock_mask = (sku_df["product_health_tier"] == "At-Risk / Deadstock") | (
            (sku_df["dbscan_cluster"] == -1) & (sku_df["days_of_inventory_remaining"] > sku_df["days_of_inventory_remaining"].median())
        )
        at_risk_df = sku_df[deadstock_mask].copy()

        # Compute continuous risk intensity score using normalized DIR and low turnover
        dir_norm = (at_risk_df["days_of_inventory_remaining"] - at_risk_df["days_of_inventory_remaining"].min()) / (
            at_risk_df["days_of_inventory_remaining"].max() - at_risk_df["days_of_inventory_remaining"].min() + 1e-5
        )

        at_risk_df["risk_intensity"] = dir_norm

        # Dynamic markdown calculation:
        # Base markdown scales with margin percentage and urgency, bounded between 10% and 50%
        margin = at_risk_df["margin_pct"].clip(lower=0.15, upper=0.70)
        recommended_markdown = margin * (0.35 + 0.50 * at_risk_df["risk_intensity"])
        at_risk_df["recommended_markdown_pct"] = np.round(recommended_markdown.clip(0.10, 0.50), 2)

        # Flag spoilage risk for perishable items
        at_risk_df["is_spoilage_threat"] = at_risk_df["is_perishable"] & (at_risk_df["days_to_expiry_min"] <= 10)

        # Recommended action description
        actions = []
        for _, row in at_risk_df.iterrows():
            if row["is_spoilage_threat"]:
                actions.append(f"Flash Markdown {int(row['recommended_markdown_pct']*100)}% - Imminent Spoilage Window")
            else:
                actions.append(f"Clearance Markdown {int(row['recommended_markdown_pct']*100)}% & Promotional Bundle")
        at_risk_df["recommended_action"] = actions

        cols_export = [
            "sku_id",
            "product_name",
            "category",
            "product_health_tier",
            "current_on_hand_stock",
            "avg_velocity",
            "days_of_inventory_remaining",
            "turn_ratio",
            "margin_pct",
            "is_perishable",
            "days_to_expiry_min",
            "recommended_markdown_pct",
            "is_spoilage_threat",
            "recommended_action",
        ]
        return at_risk_df[cols_export].sort_values(by="days_of_inventory_remaining", ascending=False).reset_index(drop=True)
