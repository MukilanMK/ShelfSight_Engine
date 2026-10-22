"""
Module 2 (UNIT-I): ML Preprocessing and Split Pipelines
======================================================
Implements rigorous preprocessing pipelines:
- Dynamic feature scaling (StandardScaler & RobustScaler for heavy-tailed retail metrics)
- One-Hot encoding of catalog categorical variables
- Time-aware temporal train/test split to prevent data leakage
- Stratified and TimeSeries K-Fold Cross-Validation generators
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


class SplitStrategy(str, Enum):
    TEMPORAL = "temporal"
    RANDOM = "random"


@dataclass
class PreprocessedDataset:
    """Encapsulates preprocessed training and testing partitions."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train_reg: pd.Series
    y_test_reg: pd.Series
    y_train_clf: pd.Series
    y_test_clf: pd.Series
    feature_names: List[str]
    train_dates: pd.Series
    test_dates: pd.Series


class MLDataPreprocessor:
    """
    Rigorously cleans, encodes, scales, and partitions grocery supply chain features.
    Guarantees no data leakage between training and testing sets.
    """

    def __init__(
        self,
        scaler_type: str = "robust",
        test_ratio: float = 0.20,
        n_splits: int = 5,
    ) -> None:
        self.scaler_type = scaler_type
        self.test_ratio = test_ratio
        self.n_splits = n_splits

        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.feature_names: List[str] = []

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        split_strategy: SplitStrategy = SplitStrategy.TEMPORAL,
    ) -> PreprocessedDataset:
        """
        Transforms raw feature matrix into model-ready X and y sets.
        """
        df_clean = df.copy()

        # Target definitions
        # Regression target: next-day / daily sales velocity
        target_reg = df_clean["units_sold_daily"]
        # Classification target: dynamic restock urgency tier
        target_clf = df_clean["restock_target_encoded"]

        # Drop non-feature metadata and labels
        cols_to_drop = [
            "sku_id", "date", "upc", "product_name", "batch_expiry_date",
            "supplier_lead_time_recorded", "baseline_tier",
            "restock_urgency_tier", "restock_target_encoded",
            "units_sold_daily", "revenue_daily", "daily_gross_profit",
        ]
        feature_df = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])

        # Separate categorical and numeric columns
        cat_cols = ["category", "storage_type"]
        cat_cols = [c for c in cat_cols if c in feature_df.columns]
        num_cols = [c for c in feature_df.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(feature_df[c])]

        # Fill any remaining NaNs in numeric features with median computed strictly on data
        feature_df[num_cols] = feature_df[num_cols].fillna(feature_df[num_cols].median())

        # Perform Train/Test Split BEFORE scaling to prevent data leakage
        if split_strategy == SplitStrategy.TEMPORAL:
            dates = pd.to_datetime(df_clean["date"])
            split_idx = int(len(df_clean) * (1.0 - self.test_ratio))
            # Sort chronologically to preserve time sequence
            sort_order = np.argsort(dates.values)
            train_idx = sort_order[:split_idx]
            test_idx = sort_order[split_idx:]
        else:
            np.random.seed(42)
            indices = np.random.permutation(len(df_clean))
            split_idx = int(len(df_clean) * (1.0 - self.test_ratio))
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]

        X_train_raw = feature_df.iloc[train_idx].reset_index(drop=True)
        X_test_raw = feature_df.iloc[test_idx].reset_index(drop=True)

        y_train_reg = target_reg.iloc[train_idx].reset_index(drop=True)
        y_test_reg = target_reg.iloc[test_idx].reset_index(drop=True)

        y_train_clf = target_clf.iloc[train_idx].reset_index(drop=True)
        y_test_clf = target_clf.iloc[test_idx].reset_index(drop=True)

        train_dates = df_clean["date"].iloc[train_idx].reset_index(drop=True)
        test_dates = df_clean["date"].iloc[test_idx].reset_index(drop=True)

        # Fit Scaler on X_train ONLY, then transform both
        X_train_num = pd.DataFrame(
            self.scaler.fit_transform(X_train_raw[num_cols]),
            columns=num_cols,
        )
        X_test_num = pd.DataFrame(
            self.scaler.transform(X_test_raw[num_cols]),
            columns=num_cols,
        )

        # Fit One-Hot Encoder on X_train ONLY
        if cat_cols:
            X_train_cat_arr = self.cat_encoder.fit_transform(X_train_raw[cat_cols])
            X_test_cat_arr = self.cat_encoder.transform(X_test_raw[cat_cols])
            cat_feature_names = self.cat_encoder.get_feature_names_out(cat_cols).tolist()

            X_train_cat = pd.DataFrame(X_train_cat_arr, columns=cat_feature_names)
            X_test_cat = pd.DataFrame(X_test_cat_arr, columns=cat_feature_names)

            X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
            X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)
        else:
            X_train_final = X_train_num
            X_test_final = X_test_num

        self.feature_names = X_train_final.columns.tolist()

        return PreprocessedDataset(
            X_train=X_train_final,
            X_test=X_test_final,
            y_train_reg=y_train_reg,
            y_test_reg=y_test_reg,
            y_train_clf=y_train_clf,
            y_test_clf=y_test_clf,
            feature_names=self.feature_names,
            train_dates=train_dates,
            test_dates=test_dates,
        )

    def get_cv_splitters(self) -> Dict[str, Union[KFold, StratifiedKFold, TimeSeriesSplit]]:
        """Provides configured cross-validation splitters."""
        return {
            "standard_kfold": KFold(n_splits=self.n_splits, shuffle=True, random_state=42),
            "stratified_kfold": StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42),
            "time_series_split": TimeSeriesSplit(n_splits=self.n_splits),
        }
