"""
Module 3 (UNIT-II): Multi-Method Feature Selection Framework
============================================================
Implements syllabus requirements for rigorous feature reduction:
1. Filter Methods:
   - Univariate Selection (SelectKBest with ANOVA F-test / f_regression)
   - Tree-based Random Forest Feature Importance (MDI)
2. Wrapper Methods:
   - Recursive Feature Elimination (RFE) with Cross-Validation
3. Embedded Methods:
   - LASSO Regularization with L1 Penalty to prune zero-weight coefficients
4. Multi-Method Consensus Scoring:
   - Aggregates rankings across all 3 paradigms to output top predictors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LassoCV


@dataclass
class FeatureSelectionSummary:
    """Summary rankings and consensus scores across all feature selection methods."""
    ranking_table: pd.DataFrame
    selected_features: List[str]
    filter_top_k: List[str]
    rf_top_k: List[str]
    rfe_selected: List[str]
    lasso_active_features: List[str]


class FeatureSelector:
    """
    Applies Filter, Wrapper, and Embedded feature selection algorithms to identify
    the most predictive retail metrics while discarding collinear and noisy features.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        top_k: int = 15,
    ) -> FeatureSelectionSummary:
        """
        Runs Filter, Wrapper, and Embedded selection methods on regression target
        and derives a consensus ranking of top predictors.
        """
        feature_names = X_train.columns.tolist()
        num_features = len(feature_names)
        k = min(top_k, num_features)

        print(f"[FeatureSelector] Evaluating {num_features} candidate features across selection paradigms...")

        # 1. FILTER METHOD A: Univariate Selection (SelectKBest with f_regression)
        print("[FeatureSelector] 1. Filter: Computing Univariate F-scores (SelectKBest)...")
        selector_kbest = SelectKBest(score_func=f_regression, k=k)
        selector_kbest.fit(X_train, y_train)
        f_scores = np.nan_to_num(selector_kbest.scores_, nan=0.0)
        kbest_indices = np.argsort(f_scores)[::-1]
        kbest_ranks = np.empty_like(kbest_indices)
        kbest_ranks[kbest_indices] = np.arange(1, num_features + 1)
        filter_top_features = [feature_names[i] for i in kbest_indices[:k]]

        # 1. FILTER METHOD B: Random Forest Feature Importance (MDI)
        print("[FeatureSelector] 2. Filter: Computing Random Forest Feature Importances...")
        rf_model = RandomForestRegressor(
            n_estimators=60,
            max_depth=8,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        rf_importances = rf_model.feature_importances_
        rf_indices = np.argsort(rf_importances)[::-1]
        rf_ranks = np.empty_like(rf_indices)
        rf_ranks[rf_indices] = np.arange(1, num_features + 1)
        rf_top_features = [feature_names[i] for i in rf_indices[:k]]

        # 2. WRAPPER METHOD: Recursive Feature Elimination (RFE)
        print(f"[FeatureSelector] 3. Wrapper: Running Recursive Feature Elimination (RFE to {k} features)...")
        base_estimator = RandomForestRegressor(
            n_estimators=30,
            max_depth=6,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rfe = RFE(estimator=base_estimator, n_features_to_select=k, step=2)
        rfe.fit(X_train, y_train)
        rfe_ranks = rfe.ranking_
        rfe_selected_features = [f for f, support in zip(feature_names, rfe.support_) if support]

        # 3. EMBEDDED METHOD: LASSO Regularization with L1 penalty
        print("[FeatureSelector] 4. Embedded: Fitting LassoCV with L1 sparsity regularization...")
        lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=2000)
        lasso.fit(X_train, y_train)
        lasso_coefs = np.abs(lasso.coef_)
        lasso_active = [f for f, coef in zip(feature_names, lasso_coefs) if coef > 1e-4]
        # Invert coef magnitudes for ranking: largest abs coef gets rank 1
        lasso_indices = np.argsort(lasso_coefs)[::-1]
        lasso_ranks = np.empty_like(lasso_indices)
        lasso_ranks[lasso_indices] = np.arange(1, num_features + 1)

        # 4. COMPOSITE CONSENSUS SCORING
        # Average rank across all 4 metrics (lower score = higher importance)
        composite_rank = (kbest_ranks + rf_ranks + rfe_ranks + lasso_ranks) / 4.0

        ranking_df = pd.DataFrame({
            "feature": feature_names,
            "f_score": f_scores,
            "kbest_rank": kbest_ranks,
            "rf_importance": rf_importances,
            "rf_rank": rf_ranks,
            "rfe_rank": rfe_ranks,
            "lasso_coef": lasso_coefs,
            "lasso_rank": lasso_ranks,
            "composite_rank_score": composite_rank,
        }).sort_values(by="composite_rank_score", ascending=True).reset_index(drop=True)

        consensus_selected = ranking_df["feature"].head(k).tolist()

        print(f"[FeatureSelector] Consensus isolated top {len(consensus_selected)} features:")
        for idx, feat in enumerate(consensus_selected[:8], 1):
            print(f"  {idx}. {feat}")

        return FeatureSelectionSummary(
            ranking_table=ranking_df,
            selected_features=consensus_selected,
            filter_top_k=filter_top_features,
            rf_top_k=rf_top_features,
            rfe_selected=rfe_selected_features,
            lasso_active_features=lasso_active,
        )
