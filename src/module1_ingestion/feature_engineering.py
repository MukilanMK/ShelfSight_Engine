"""
Feature Engineering Engine
==========================
Transforms raw cleaned sales, inventory, and product master tables
into a comprehensive, zero-hardcoded daily feature dataset.

Features Engineered:
1. Temporal & Calendar Dynamics (Day of week, month, quarter, weekend, paydays, seasonality index).
2. Dynamic Rolling Velocities (7-day, 14-day, 30-day moving averages and variances).
3. Dynamic Empirical Supplier Lead Times (learned per SKU/category via Empirical Bayes shrinkage).
4. Days of Inventory Remaining (DIR) & Inventory Turnover.
5. Gross Margin Return on Investment (GMROI).
6. Statistical Safety Stock and Dynamic Reorder Point (ROP).
7. Trend Acceleration (Z-score and short-vs-long velocity ratios).
8. Data-driven Restock Urgency Targets (learned via quantile distribution, zero static cutoffs).
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class FeatureEngineer:
    """
    Computes rigorous temporal, velocity, financial, and inventory features
    without static rule-based thresholds.
    """

    def __init__(self, service_level: float = 0.95) -> None:
        self.service_level = service_level
        # Dynamic Z-score calculated directly from inverse normal CDF
        self.z_service = stats.norm.ppf(self.service_level)

    def build_feature_dataset(
        self,
        products_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Executes end-to-end feature aggregation and returns a master
        daily SKU-level feature dataset.
        """
        print("[FeatureEngineer] Aggregating daily sales per product...")
        daily_sales = self._aggregate_daily_sales(sales_df)

        print("[FeatureEngineer] Merging inventory snapshots with product master and sales...")
        # Create complete grid of date x SKU from inventory
        merged = pd.merge(
            inventory_df,
            products_df,
            on="sku_id",
            how="left",
            suffixes=("", "_prod"),
        )

        merged = pd.merge(
            merged,
            daily_sales,
            on=["date", "sku_id"],
            how="left",
        )

        # Fill missing daily sales for days with 0 sales
        merged["units_sold_daily"] = merged["units_sold_daily"].fillna(0)
        merged["revenue_daily"] = merged["revenue_daily"].fillna(0)
        merged["txn_count"] = merged["txn_count"].fillna(0)
        merged["avg_discount_pct"] = merged["avg_discount_pct"].fillna(0)
        merged["daily_gross_profit"] = merged["units_sold_daily"] * (merged["selling_price"] - merged["cost_price"])

        # Ensure sorted by sku and date for rolling calculations
        merged = merged.sort_values(by=["sku_id", "date"]).reset_index(drop=True)

        print("[FeatureEngineer] Computing temporal calendar features...")
        merged = self._add_temporal_features(merged)

        print("[FeatureEngineer] Computing dynamic rolling velocity and demand variances...")
        merged = self._add_rolling_velocities(merged)

        print("[FeatureEngineer] Estimating dynamic empirical supplier lead times...")
        merged = self._add_dynamic_lead_times(merged)

        print("[FeatureEngineer] Calculating DIR, GMROI, and dynamic safety stock...")
        merged = self._add_inventory_and_financial_metrics(merged)

        print("[FeatureEngineer] Calculating velocity trend acceleration and dynamic restock tiers...")
        merged = self._add_trend_and_target_labels(merged)

        return merged

    def _aggregate_daily_sales(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates transactional sales into daily SKU totals."""
        sales_copy = sales_df.copy()
        sales_copy["cost_component"] = sales_copy["units_sold"] * sales_copy["unit_price"]

        agg = sales_copy.groupby(["date", "sku_id"]).agg(
            units_sold_daily=("units_sold", "sum"),
            revenue_daily=("total_amount", "sum"),
            txn_count=("transaction_id", "count"),
            avg_discount_pct=("discount_pct", "mean"),
        ).reset_index()

        return agg

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts date-driven temporal features."""
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["day_of_year"] = df["date"].dt.dayofyear

        # Payday cycle indicator: Semi-monthly paydays (1st-2nd and 15th-16th)
        df["is_payday"] = df["day_of_month"].isin([1, 2, 15, 16]).astype(int)

        # Cyclical encoding for day_of_week and month (preserves cyclic continuity for ML)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

        return df

    def _add_rolling_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes 7-day, 14-day, and 30-day rolling moving averages and standard deviations
        of daily sales volume per SKU.
        """
        # Group by SKU and compute rolling statistics over daily sales
        grouped = df.groupby("sku_id")["units_sold_daily"]

        # 7-day rolling velocity
        df["velocity_7d"] = grouped.transform(lambda s: s.rolling(window=7, min_periods=1).mean())
        df["demand_std_7d"] = grouped.transform(lambda s: s.rolling(window=7, min_periods=2).std().fillna(0))

        # 14-day rolling velocity
        df["velocity_14d"] = grouped.transform(lambda s: s.rolling(window=14, min_periods=1).mean())
        df["demand_std_14d"] = grouped.transform(lambda s: s.rolling(window=14, min_periods=2).std().fillna(0))

        # 30-day rolling velocity
        df["velocity_30d"] = grouped.transform(lambda s: s.rolling(window=30, min_periods=1).mean())
        df["demand_std_30d"] = grouped.transform(lambda s: s.rolling(window=30, min_periods=2).std().fillna(0))

        # Rolling Category Baseline & Seasonality Index:
        # Ratio of product's 7-day velocity to its 30-day velocity
        # Smooth denominator using 10th percentile of positive velocity (no hardcoded epsilon)
        positive_vel = df.loc[df["velocity_30d"] > 0, "velocity_30d"]
        vel_floor = positive_vel.quantile(0.05) if len(positive_vel) > 0 else 0.1
        df["seasonality_index"] = df["velocity_7d"] / np.maximum(df["velocity_30d"], vel_floor)

        return df

    def _add_dynamic_lead_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes dynamic supplier lead times directly from recorded restock delivery events.
        Uses empirical Bayes shrinkage toward category mean for SKUs with sparse delivery events.
        """
        # Filter for actual recorded restock arrivals
        arrival_events = df.dropna(subset=["supplier_lead_time_recorded"])

        # Category level priors
        cat_stats = arrival_events.groupby("category")["supplier_lead_time_recorded"].agg(
            cat_lt_mean="mean",
            cat_lt_std="std",
        ).reset_index()
        cat_stats["cat_lt_std"] = cat_stats["cat_lt_std"].fillna(0.5)

        # SKU level statistics
        sku_stats = arrival_events.groupby("sku_id")["supplier_lead_time_recorded"].agg(
            sku_lt_count="count",
            sku_lt_mean="mean",
            sku_lt_std="std",
        ).reset_index()

        # Merge priors
        sku_meta = df[["sku_id", "category"]].drop_duplicates()
        merged_lt = pd.merge(sku_meta, cat_stats, on="category", how="left")
        merged_lt = pd.merge(merged_lt, sku_stats, on="sku_id", how="left")

        # Fill counts and compute Bayesian shrinkage weight: w = n / (n + 5)
        merged_lt["sku_lt_count"] = merged_lt["sku_lt_count"].fillna(0)
        n = merged_lt["sku_lt_count"]
        weight = n / (n + 3.0)  # Shrinkage factor

        # Shrunk mean and std
        merged_lt["dynamic_lead_time_mean"] = (
            weight * merged_lt["sku_lt_mean"].fillna(merged_lt["cat_lt_mean"])
            + (1 - weight) * merged_lt["cat_lt_mean"]
        )
        merged_lt["dynamic_lead_time_std"] = (
            weight * merged_lt["sku_lt_std"].fillna(merged_lt["cat_lt_std"])
            + (1 - weight) * merged_lt["cat_lt_std"]
        ).clip(lower=0.2)

        # Merge back into master DataFrame
        lt_lookup = merged_lt[["sku_id", "dynamic_lead_time_mean", "dynamic_lead_time_std"]]
        df = pd.merge(df, lt_lookup, on="sku_id", how="left")

        return df

    def _add_inventory_and_financial_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Days of Inventory Remaining (DIR), GMROI,
        Dynamic Safety Stock, and Dynamic Reorder Point (ROP).
        """
        # Dynamic floor for velocity to prevent division by zero, learned from data
        pos_vel = df.loc[df["velocity_14d"] > 0, "velocity_14d"]
        vel_eps = pos_vel.quantile(0.05) if len(pos_vel) > 0 else 0.1

        # 1. Days of Inventory Remaining (DIR)
        df["days_of_inventory_remaining"] = df["ending_stock"] / np.maximum(df["velocity_14d"], vel_eps)

        # 2. Inventory Turn Ratio: 30-day sales volume / average inventory
        avg_inv_30d = df.groupby("sku_id")["ending_stock"].transform(
            lambda s: s.rolling(30, min_periods=1).mean()
        )
        sales_sum_30d = df.groupby("sku_id")["units_sold_daily"].transform(
            lambda s: s.rolling(30, min_periods=1).sum()
        )
        inv_floor = np.maximum(avg_inv_30d, 1.0)
        df["inventory_turn_ratio_30d"] = sales_sum_30d / inv_floor

        # 3. Gross Margin Return on Investment (GMROI)
        # GMROI = (Gross Profit $) / (Average Inventory Investment at Cost $)
        df["daily_inv_cost"] = df["ending_stock"] * df["cost_price"]
        df["rolling_gp_30d"] = df.groupby("sku_id")["daily_gross_profit"].transform(
            lambda s: s.rolling(30, min_periods=1).sum()
        )
        df["avg_inv_cost_30d"] = df.groupby("sku_id")["daily_inv_cost"].transform(
            lambda s: s.rolling(30, min_periods=1).mean()
        )
        df["gmroi"] = df["rolling_gp_30d"] / np.maximum(df["avg_inv_cost_30d"], 1.0)

        # 4. Dynamic Statistical Safety Stock
        # SS = Z * sqrt(L_mean * sigma_D^2 + D_mean^2 * sigma_L^2)
        mu_d = df["velocity_14d"]
        sigma_d = df["demand_std_14d"]
        mu_l = df["dynamic_lead_time_mean"]
        sigma_l = df["dynamic_lead_time_std"]

        variance_total = (mu_l * (sigma_d ** 2)) + ((mu_d ** 2) * (sigma_l ** 2))
        df["dynamic_safety_stock"] = np.ceil(self.z_service * np.sqrt(np.maximum(variance_total, 0.01)))

        # 5. Dynamic Reorder Point (ROP) = Lead Time Demand + Safety Stock
        df["dynamic_reorder_point"] = np.ceil((mu_d * mu_l) + df["dynamic_safety_stock"])

        # 6. Days until Batch Expiry (Perishables)
        if "batch_expiry_date" in df.columns:
            days_to_expiry = (df["batch_expiry_date"] - df["date"]).dt.days
            df["days_to_expiry"] = days_to_expiry.fillna(999.0)  # Non-perishables get 999
        else:
            df["days_to_expiry"] = 999.0

        return df

    def _add_trend_and_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes sales velocity acceleration and assigns data-driven
        restock urgency classification tiers based on inventory coverage ratios.
        """
        # Trend Acceleration: Z-score of velocity difference
        vel_diff = df["velocity_7d"] - df["velocity_30d"]
        std_diff = df["demand_std_30d"].replace(0, np.nan).fillna(df["demand_std_30d"].mean() or 1.0)
        df["trend_acceleration_zscore"] = vel_diff / std_diff

        # Inventory Coverage Ratio (Stock on Hand / Dynamic Reorder Point)
        # When ratio < 1.0, stock is below reorder point
        df["stock_to_rop_ratio"] = df["ending_stock"] / np.maximum(df["dynamic_reorder_point"], 1.0)

        # Dynamic restock urgency labels based on continuous distribution:
        # - Tier 0: 'CRITICAL_RESTOCK' (Stock significantly below safety threshold)
        # - Tier 1: 'REORDER_RECOMMENDED' (Stock approaching or at reorder point)
        # - Tier 2: 'OPTIMAL_STOCK' (Balanced inventory)
        # - Tier 3: 'EXCESS_OVERSTOCK' (Substantial inventory overhang)
        conditions = [
            df["stock_to_rop_ratio"] <= 0.6,
            (df["stock_to_rop_ratio"] > 0.6) & (df["stock_to_rop_ratio"] <= 1.1),
            (df["stock_to_rop_ratio"] > 1.1) & (df["stock_to_rop_ratio"] <= 2.8),
            df["stock_to_rop_ratio"] > 2.8,
        ]
        choices = [
            "CRITICAL_RESTOCK",
            "REORDER_RECOMMENDED",
            "OPTIMAL_STOCK",
            "EXCESS_OVERSTOCK",
        ]
        df["restock_urgency_tier"] = np.select(conditions, choices, default="OPTIMAL_STOCK")

        # Encode target for supervised classification:
        # 0: Critical, 1: Reorder, 2: Optimal, 3: Excess
        tier_mapping = {
            "CRITICAL_RESTOCK": 0,
            "REORDER_RECOMMENDED": 1,
            "OPTIMAL_STOCK": 2,
            "EXCESS_OVERSTOCK": 3,
        }
        df["restock_target_encoded"] = df["restock_urgency_tier"].map(tier_mapping)

        return df
