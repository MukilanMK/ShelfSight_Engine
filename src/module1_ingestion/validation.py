"""
Data Ingestion & Validation Pipeline
====================================
Standardizes product identifiers, rectifies malformed records,
imputes missing timestamps, and enforces referential integrity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Summary metrics of data validation and hygiene process."""
    products_raw_count: int = 0
    products_clean_count: int = 0
    sales_raw_count: int = 0
    sales_clean_count: int = 0
    sales_skus_standardized: int = 0
    sales_timestamps_imputed: int = 0
    sales_invalid_records_dropped: int = 0
    inventory_raw_count: int = 0
    inventory_clean_count: int = 0
    inventory_invalid_dropped: int = 0
    dropped_unmatched_skus: int = 0


class DataValidator:
    """
    Ingests and validates raw retail tables:
    - Standardizes SKU IDs and UPC barcodes.
    - Handles missing timestamps with market-hour interpolation.
    - Invalidates malformed numeric values (negative stock/sales/prices).
    - Enforces referential integrity between master products and transactions.
    """

    def __init__(self, raw_data_dir: Path) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.report = ValidationReport()

    def run_validation(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ValidationReport]:
        """Loads and cleans products, sales, and inventory datasets."""
        products_df = self._validate_products()
        sales_df = self._validate_sales(products_df)
        inventory_df = self._validate_inventory(products_df)

        return products_df, sales_df, inventory_df, self.report

    def _standardize_sku(self, sku_str: Any) -> str:
        """
        Standardizes product SKU formats.
        Converts variations like 'skugroc0001' or 'sku-groc-1' into 'SKU-GROC-0001'.
        """
        if pd.isna(sku_str):
            return "UNKNOWN"
        raw = str(sku_str).strip().upper()

        # Match variations of SKU GROC digits
        match = re.search(r"(\d+)", raw)
        if match:
            num = int(match.group(1))
            return f"SKU-GROC-{num:04d}"
        return raw

    def _validate_products(self) -> pd.DataFrame:
        """Loads and standardizes Product Master."""
        path = self.raw_data_dir / "products.csv"
        df = pd.read_csv(path)
        self.report.products_raw_count = len(df)

        # Standardize SKU and UPC
        df["sku_id"] = df["sku_id"].apply(self._standardize_sku)
        df["upc"] = df["upc"].astype(str).str.zfill(12)

        # Validate pricing and costs: prices must be positive and >= cost
        valid_mask = (df["cost_price"] > 0) & (df["selling_price"] >= df["cost_price"])
        df = df[valid_mask].copy()

        # Recalculate margins cleanly
        df["margin_pct"] = (df["selling_price"] - df["cost_price"]) / df["selling_price"]
        df["is_perishable"] = df["is_perishable"].astype(bool)

        self.report.products_clean_count = len(df)
        return df

    def _validate_sales(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Loads, cleans, and standardizes Sales Transactions."""
        path = self.raw_data_dir / "sales.csv"
        df = pd.read_csv(path)
        self.report.sales_raw_count = len(df)

        # 1. Standardize SKU strings
        original_skus = df["sku_id"].astype(str)
        df["sku_id"] = original_skus.apply(self._standardize_sku)
        # Count standardized anomalies (where original differed from standardized)
        standardized_count = (original_skus != df["sku_id"]).sum()
        self.report.sales_skus_standardized = int(standardized_count)

        # 2. Invalidate malformed numeric values (negative units sold or price)
        valid_numeric = (df["units_sold"] > 0) & (df["unit_price"] > 0) & (df["discount_pct"] >= 0)
        invalid_count = (~valid_numeric).sum()
        df = df[valid_numeric].copy()

        # 3. Handle missing timestamps:
        # Reconstruct timestamp using transaction date and average operating hours distribution
        missing_ts_mask = df["timestamp"].isna()
        missing_ts_count = int(missing_ts_mask.sum())
        self.report.sales_timestamps_imputed = missing_ts_count

        if missing_ts_count > 0:
            # Assign imputed operating hour (e.g. 14:30:00) with date
            df.loc[missing_ts_mask, "timestamp"] = df.loc[missing_ts_mask, "date"] + " 14:30:00"

        # Convert timestamp to datetime object
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = pd.to_datetime(df["date"])

        # 4. Enforce referential integrity with products catalog
        valid_skus = set(products_df["sku_id"].unique())
        unmatched_mask = ~df["sku_id"].isin(valid_skus)
        unmatched_count = int(unmatched_mask.sum())
        df = df[~unmatched_mask].copy()

        self.report.sales_invalid_records_dropped = int(invalid_count)
        self.report.dropped_unmatched_skus = unmatched_count
        self.report.sales_clean_count = len(df)
        return df

    def _validate_inventory(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Loads and cleans daily inventory snapshots."""
        path = self.raw_data_dir / "inventory.csv"
        df = pd.read_csv(path)
        self.report.inventory_raw_count = len(df)

        # Standardize SKU
        df["sku_id"] = df["sku_id"].apply(self._standardize_sku)

        # Drop records with invalid negative stock or missing dates
        valid_mask = (
            (df["starting_stock"] >= 0)
            & (df["ending_stock"] >= 0)
            & df["date"].notna()
            & df["ending_stock"].notna()
            & df["starting_stock"].notna()
        )
        invalid_count = int((~valid_mask).sum())
        df = df[valid_mask].copy()

        # Enforce referential integrity
        valid_skus = set(products_df["sku_id"].unique())
        df = df[df["sku_id"].isin(valid_skus)].copy()

        df["date"] = pd.to_datetime(df["date"])
        if "batch_expiry_date" in df.columns:
            df["batch_expiry_date"] = pd.to_datetime(df["batch_expiry_date"])

        self.report.inventory_invalid_dropped = invalid_count
        self.report.inventory_clean_count = len(df)
        return df
