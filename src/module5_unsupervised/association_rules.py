"""
Module 5 (UNIT-IV): Association Rule Mining for Dynamic Retail Bundling
======================================================================
Applies the Apriori algorithm to transaction market baskets to:
1. Discover high-confidence, high-lift item co-purchases.
2. Strategically pair Deadstock and Slow-moving products with Fast-moving staples
   to accelerate inventory liquidation through co-promotions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


@dataclass
class AssociationMiningResults:
    """Encapsulates discovered rules, frequent itemsets, and strategic bundles."""
    frequent_itemsets: pd.DataFrame
    all_rules: pd.DataFrame
    slow_mover_bundles: pd.DataFrame
    min_support_used: float


class MarketBasketAnalyzer:
    """
    Executes Apriori association rule mining on grocery market baskets.
    """

    def __init__(
        self,
        min_support: float = 0.015,
        min_lift: float = 1.20,
        min_confidence: float = 0.25,
    ) -> None:
        self.min_support = min_support
        self.min_lift = min_lift
        self.min_confidence = min_confidence

    def mine_rules(
        self,
        sales_df: pd.DataFrame,
        products_df: pd.DataFrame,
        deadstock_skus: Optional[Set[str]] = None,
    ) -> AssociationMiningResults:
        """
        Transforms raw transaction stream into basket matrix and mines association rules.
        """
        print("[MarketBasketAnalyzer] Preparing market basket matrix...")
        # Map sku_id to readable product name
        sku_to_name = dict(zip(products_df["sku_id"], products_df["product_name"]))
        sku_to_tier = dict(zip(products_df["sku_id"], products_df.get("baseline_tier", products_df["category"])))

        sales_with_names = sales_df[["basket_id", "sku_id"]].drop_duplicates().copy()
        sales_with_names["product_name"] = sales_with_names["sku_id"].map(sku_to_name)

        # Filter out baskets with only 1 item since rules require pairs
        basket_counts = sales_with_names.groupby("basket_id").size()
        multi_item_baskets = basket_counts[basket_counts >= 2].index
        sales_multi = sales_with_names[sales_with_names["basket_id"].isin(multi_item_baskets)]

        print(f"[MarketBasketAnalyzer] Transforming {len(multi_item_baskets)} multi-item shopping baskets...")
        # Pivot into boolean matrix: baskets x products
        basket_matrix = (
            sales_multi.groupby(["basket_id", "product_name"])["sku_id"]
            .count()
            .unstack()
            .fillna(0)
            .astype(bool)
        )

        # 1. Run Apriori Algorithm (Dynamic Support Fallback if sparse)
        print(f"[MarketBasketAnalyzer] Running Apriori (target min_support={self.min_support})...")
        support_levels = [self.min_support, 0.010, 0.005]
        frequent_itemsets = pd.DataFrame()
        effective_support = self.min_support

        for sup in support_levels:
            frequent = apriori(basket_matrix, min_support=sup, use_colnames=True)
            if len(frequent) >= 8:
                frequent_itemsets = frequent
                effective_support = sup
                break

        if frequent_itemsets.empty:
            frequent_itemsets = apriori(basket_matrix, min_support=0.003, use_colnames=True)
            effective_support = 0.003

        print(f"[MarketBasketAnalyzer] Discovered {len(frequent_itemsets)} frequent itemsets at support >= {effective_support:.3f}")

        # 2. Extract Association Rules
        print(f"[MarketBasketAnalyzer] Generating association rules (min_lift={self.min_lift}, min_confidence={self.min_confidence})...")
        rules = association_rules(
            frequent_itemsets,
            metric="lift",
            min_threshold=self.min_lift,
        )
        rules = rules[rules["confidence"] >= self.min_confidence].copy()

        # Format string antecedents and consequents
        rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(list(s)))
        rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(list(s)))

        # Round metric columns
        for col in ["support", "confidence", "lift", "leverage", "conviction"]:
            if col in rules.columns:
                rules[col] = rules[col].round(4)

        rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

        # 3. Strategic Slow-Mover & Deadstock Cross-Selling Bundles
        print("[MarketBasketAnalyzer] Identifying strategic cross-selling bundles for inventory acceleration...")
        name_to_sku = dict(zip(products_df["product_name"], products_df["sku_id"]))
        deadstock_set = deadstock_skus or set()

        bundle_candidates = []
        for _, rule in rules.iterrows():
            ant_items = list(rule["antecedents"])
            con_items = list(rule["consequents"])

            # Check if any item in the rule belongs to deadstock/slow-moving group
            ant_skus = [name_to_sku.get(item) for item in ant_items]
            con_skus = [name_to_sku.get(item) for item in con_items]

            has_slow_ant = any(s in deadstock_set for s in ant_skus)
            has_slow_con = any(s in deadstock_set for s in con_skus)

            # High synergy bundle
            if rule["lift"] >= 1.3:
                bundle_candidates.append({
                    "primary_item": rule["antecedents_str"],
                    "bundled_recommendation": rule["consequents_str"],
                    "support": rule["support"],
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                    "is_deadstock_liquidation_bundle": has_slow_ant or has_slow_con,
                    "bundle_strategy": "Bundle & Save Promo: Pair with high-velocity anchor to clear at-risk inventory"
                    if (has_slow_ant or has_slow_con)
                    else "Standard Basket Cross-Sell",
                })

        bundles_df = pd.DataFrame(bundle_candidates)
        if not bundles_df.empty:
            bundles_df = bundles_df.sort_values(by=["is_deadstock_liquidation_bundle", "lift"], ascending=[False, False]).reset_index(drop=True)

        print(f"[MarketBasketAnalyzer] Extracted {len(rules)} association rules and {len(bundles_df)} strategic bundle opportunities.")

        return AssociationMiningResults(
            frequent_itemsets=frequent_itemsets,
            all_rules=rules,
            slow_mover_bundles=bundles_df,
            min_support_used=effective_support,
        )
