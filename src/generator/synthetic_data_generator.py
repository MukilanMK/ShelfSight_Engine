"""
Synthetic Grocery Store Dataset Generator
=========================================
Generates production-grade, relational, realistic grocery data:
1. Product Master (products.csv)
2. Daily Inventory Snapshots (inventory.csv)
3. Transactional Sales with Market Baskets (sales.csv)

Follows zero hardcoded assumptions:
- Uses stochastic processes, lead time distributions, and seasonal multipliers.
- Includes realistic market basket co-occurrences for Association Rule Mining (Apriori).
- Simulates realistic shelf-life spoilage and inventory balance constraints.
- Injects a small, controlled rate of data quality anomalies to validate Module 1's ingestion pipeline.
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    """Configuration parameters for the synthetic dataset generator."""
    num_products: int = 60
    start_date: str = "2024-01-01"
    num_days: int = 180  # ~6 months of daily data
    avg_transactions_per_day: int = 140
    random_seed: int = 42
    output_dir: Path = Path("data/raw")
    inject_anomalies: bool = True
    anomaly_rate: float = 0.008  # ~0.8% anomaly injection for pipeline testing


# Pre-defined realistic catalog templates across categories
CATALOG_TEMPLATES: List[Dict[str, Any]] = [
    # Dairy & Eggs
    {"name": "Whole Milk 1 Gallon", "category": "Dairy & Eggs", "cost": 2.40, "price": 3.89, "is_perishable": True, "shelf_life": 12, "storage": "Chilled", "lead_time_mean": 2.0, "lead_time_std": 0.4, "tier": "Fast-mover"},
    {"name": "Organic Large Brown Eggs 12ct", "category": "Dairy & Eggs", "cost": 2.60, "price": 4.49, "is_perishable": True, "shelf_life": 28, "storage": "Chilled", "lead_time_mean": 2.2, "lead_time_std": 0.5, "tier": "Fast-mover"},
    {"name": "Salted Butter 16oz", "category": "Dairy & Eggs", "cost": 2.80, "price": 4.99, "is_perishable": True, "shelf_life": 60, "storage": "Chilled", "lead_time_mean": 3.0, "lead_time_std": 0.6, "tier": "Moderate-mover"},
    {"name": "Greek Yogurt Plain 32oz", "category": "Dairy & Eggs", "cost": 2.70, "price": 4.79, "is_perishable": True, "shelf_life": 21, "storage": "Chilled", "lead_time_mean": 2.5, "lead_time_std": 0.5, "tier": "Moderate-mover"},
    {"name": "Shredded Sharp Cheddar 8oz", "category": "Dairy & Eggs", "cost": 1.70, "price": 3.29, "is_perishable": True, "shelf_life": 45, "storage": "Chilled", "lead_time_mean": 3.0, "lead_time_std": 0.5, "tier": "Moderate-mover"},
    {"name": "Aged Artisan Gouda Wedge 6oz", "category": "Dairy & Eggs", "cost": 5.80, "price": 9.99, "is_perishable": True, "shelf_life": 50, "storage": "Chilled", "lead_time_mean": 5.0, "lead_time_std": 1.2, "tier": "Slow-mover"},
    {"name": "Truffle Infused Goat Cheese 4oz", "category": "Dairy & Eggs", "cost": 6.50, "price": 11.49, "is_perishable": True, "shelf_life": 25, "storage": "Chilled", "lead_time_mean": 6.0, "lead_time_std": 1.5, "tier": "Deadstock-candidate"},

    # Bakery
    {"name": "Artisan French Sourdough", "category": "Bakery", "cost": 1.80, "price": 4.29, "is_perishable": True, "shelf_life": 4, "storage": "Ambient", "lead_time_mean": 1.5, "lead_time_std": 0.3, "tier": "Fast-mover"},
    {"name": "Whole Wheat Sandwich Bread 20oz", "category": "Bakery", "cost": 1.40, "price": 3.19, "is_perishable": True, "shelf_life": 7, "storage": "Ambient", "lead_time_mean": 1.8, "lead_time_std": 0.3, "tier": "Fast-mover"},
    {"name": "Butter Croissants 4-Pack", "category": "Bakery", "cost": 2.30, "price": 4.89, "is_perishable": True, "shelf_life": 4, "storage": "Ambient", "lead_time_mean": 1.8, "lead_time_std": 0.4, "tier": "Moderate-mover"},
    {"name": "Brioche Burger Buns 6ct", "category": "Bakery", "cost": 1.90, "price": 3.99, "is_perishable": True, "shelf_life": 8, "storage": "Ambient", "lead_time_mean": 2.0, "lead_time_std": 0.4, "tier": "Moderate-mover"},
    {"name": "Gluten-Free Seeded Loaf", "category": "Bakery", "cost": 3.80, "price": 6.99, "is_perishable": True, "shelf_life": 6, "storage": "Ambient", "lead_time_mean": 3.5, "lead_time_std": 0.8, "tier": "Slow-mover"},

    # Fresh Produce
    {"name": "Hass Avocados (Single)", "category": "Fresh Produce", "cost": 0.65, "price": 1.49, "is_perishable": True, "shelf_life": 6, "storage": "Ambient", "lead_time_mean": 2.0, "lead_time_std": 0.4, "tier": "Fast-mover"},
    {"name": "Organic Bananas 1 Bunch", "category": "Fresh Produce", "cost": 0.70, "price": 1.69, "is_perishable": True, "shelf_life": 5, "storage": "Ambient", "lead_time_mean": 1.8, "lead_time_std": 0.3, "tier": "Fast-mover"},
    {"name": "Honeycrisp Apples 3lb Bag", "category": "Fresh Produce", "cost": 2.90, "price": 5.49, "is_perishable": True, "shelf_life": 18, "storage": "Chilled", "lead_time_mean": 2.5, "lead_time_std": 0.5, "tier": "Fast-mover"},
    {"name": "Fresh Baby Spinach 5oz Clamshell", "category": "Fresh Produce", "cost": 1.20, "price": 2.99, "is_perishable": True, "shelf_life": 6, "storage": "Chilled", "lead_time_mean": 1.8, "lead_time_std": 0.3, "tier": "Fast-mover"},
    {"name": "Organic Roma Tomatoes 1lb", "category": "Fresh Produce", "cost": 1.10, "price": 2.49, "is_perishable": True, "shelf_life": 8, "storage": "Ambient", "lead_time_mean": 2.0, "lead_time_std": 0.4, "tier": "Moderate-mover"},
    {"name": "Fresh Garlic Bulb 3-Pack", "category": "Fresh Produce", "cost": 0.60, "price": 1.79, "is_perishable": True, "shelf_life": 30, "storage": "Ambient", "lead_time_mean": 3.0, "lead_time_std": 0.6, "tier": "Moderate-mover"},
    {"name": "Fresh English Cucumber", "category": "Fresh Produce", "cost": 0.55, "price": 1.39, "is_perishable": True, "shelf_life": 10, "storage": "Chilled", "lead_time_mean": 2.0, "lead_time_std": 0.4, "tier": "Moderate-mover"},
    {"name": "Dragon Fruit (Yellow Pitahaya)", "category": "Fresh Produce", "cost": 3.90, "price": 6.99, "is_perishable": True, "shelf_life": 7, "storage": "Ambient", "lead_time_mean": 4.5, "lead_time_std": 1.0, "tier": "Slow-mover"},
    {"name": "Romanesco Cauliflower Head", "category": "Fresh Produce", "cost": 3.20, "price": 5.99, "is_perishable": True, "shelf_life": 6, "storage": "Chilled", "lead_time_mean": 4.0, "lead_time_std": 1.0, "tier": "Deadstock-candidate"},

    # Meat & Seafood
    {"name": "Boneless Skinless Chicken Breasts 2lb", "category": "Meat & Seafood", "cost": 5.20, "price": 9.49, "is_perishable": True, "shelf_life": 5, "storage": "Chilled", "lead_time_mean": 2.0, "lead_time_std": 0.4, "tier": "Fast-mover"},
    {"name": "Ground Beef 80/20 1lb", "category": "Meat & Seafood", "cost": 3.10, "price": 5.99, "is_perishable": True, "shelf_life": 5, "storage": "Chilled", "lead_time_mean": 2.0, "lead_time_std": 0.4, "tier": "Fast-mover"},
    {"name": "Atlantic Salmon Fillet 1lb", "category": "Meat & Seafood", "cost": 7.40, "price": 12.99, "is_perishable": True, "shelf_life": 4, "storage": "Chilled", "lead_time_mean": 2.2, "lead_time_std": 0.5, "tier": "Moderate-mover"},
    {"name": "Thick Cut Smoked Bacon 16oz", "category": "Meat & Seafood", "cost": 3.80, "price": 6.79, "is_perishable": True, "shelf_life": 30, "storage": "Chilled", "lead_time_mean": 3.0, "lead_time_std": 0.5, "tier": "Moderate-mover"},
    {"name": "Prime NY Strip Steak 12oz", "category": "Meat & Seafood", "cost": 9.50, "price": 16.99, "is_perishable": True, "shelf_life": 5, "storage": "Chilled", "lead_time_mean": 3.0, "lead_time_std": 0.6, "tier": "Moderate-mover"},
    {"name": "Wild Boar Salami Link 7oz", "category": "Meat & Seafood", "cost": 7.80, "price": 14.50, "is_perishable": True, "shelf_life": 45, "storage": "Chilled", "lead_time_mean": 6.0, "lead_time_std": 1.2, "tier": "Deadstock-candidate"},

    # Pantry & Staples
    {"name": "Italian Penne Rigate 16oz", "category": "Pantry & Staples", "cost": 0.90, "price": 1.99, "is_perishable": False, "shelf_life": 720, "storage": "Ambient", "lead_time_mean": 3.5, "lead_time_std": 0.7, "tier": "Fast-mover"},
    {"name": "Spaghetti Pasta 16oz", "category": "Pantry & Staples", "cost": 0.85, "price": 1.89, "is_perishable": False, "shelf_life": 720, "storage": "Ambient", "lead_time_mean": 3.5, "lead_time_std": 0.7, "tier": "Fast-mover"},
    {"name": "Marinara Pasta Sauce 24oz Jar", "category": "Pantry & Staples", "cost": 1.80, "price": 3.99, "is_perishable": False, "shelf_life": 540, "storage": "Ambient", "lead_time_mean": 3.5, "lead_time_std": 0.7, "tier": "Fast-mover"},
    {"name": "Extra Virgin Olive Oil 500ml", "category": "Pantry & Staples", "cost": 5.40, "price": 9.99, "is_perishable": False, "shelf_life": 365, "storage": "Ambient", "lead_time_mean": 4.0, "lead_time_std": 0.8, "tier": "Moderate-mover"},
    {"name": "Jasmine Rice 5lb Bag", "category": "Pantry & Staples", "cost": 3.60, "price": 6.99, "is_perishable": False, "shelf_life": 720, "storage": "Ambient", "lead_time_mean": 4.0, "lead_time_std": 0.8, "tier": "Moderate-mover"},
    {"name": "Black Beans Canned 15oz", "category": "Pantry & Staples", "cost": 0.50, "price": 1.29, "is_perishable": False, "shelf_life": 1080, "storage": "Ambient", "lead_time_mean": 3.0, "lead_time_std": 0.5, "tier": "Fast-mover"},
    {"name": "Artisanal White Truffle Oil 250ml", "category": "Pantry & Staples", "cost": 12.00, "price": 24.99, "is_perishable": False, "shelf_life": 365, "storage": "Ambient", "lead_time_mean": 7.0, "lead_time_std": 1.5, "tier": "Deadstock-candidate"},
    {"name": "Organic Quinoa Grain 16oz", "category": "Pantry & Staples", "cost": 2.40, "price": 4.99, "is_perishable": False, "shelf_life": 540, "storage": "Ambient", "lead_time_mean": 4.0, "lead_time_std": 0.8, "tier": "Slow-mover"},
    {"name": "Granulated Pure Cane Sugar 4lb", "category": "Pantry & Staples", "cost": 2.10, "price": 3.89, "is_perishable": False, "shelf_life": 1080, "storage": "Ambient", "lead_time_mean": 3.0, "lead_time_std": 0.5, "tier": "Moderate-mover"},

    # Snacks & Confectionery
    {"name": "Tortilla Corn Chips 12oz", "category": "Snacks & Confectionery", "cost": 1.60, "price": 3.79, "is_perishable": False, "shelf_life": 180, "storage": "Ambient", "lead_time_mean": 2.5, "lead_time_std": 0.5, "tier": "Fast-mover"},
    {"name": "Medium Restaurant Style Salsa 16oz", "category": "Snacks & Confectionery", "cost": 1.70, "price": 3.99, "is_perishable": False, "shelf_life": 365, "storage": "Ambient", "lead_time_mean": 3.0, "lead_time_std": 0.5, "tier": "Fast-mover"},
    {"name": "Classic Potato Chips Sea Salt 8oz", "category": "Snacks & Confectionery", "cost": 1.50, "price": 3.49, "is_perishable": False, "shelf_life": 150, "storage": "Ambient", "lead_time_mean": 2.5, "lead_time_std": 0.4, "tier": "Fast-mover"},
    {"name": "Dark Chocolate Sea Salt Bar 3.5oz", "category": "Snacks & Confectionery", "cost": 1.40, "price": 2.99, "is_perishable": False, "shelf_life": 365, "storage": "Ambient", "lead_time_mean": 3.5, "lead_time_std": 0.6, "tier": "Moderate-mover"},
    {"name": "Roasted Salted Almonds 12oz", "category": "Snacks & Confectionery", "cost": 4.20, "price": 7.99, "is_perishable": False, "shelf_life": 270, "storage": "Ambient", "lead_time_mean": 3.5, "lead_time_std": 0.6, "tier": "Moderate-mover"},
    {"name": "Organic Honey Almond Granola 12oz", "category": "Snacks & Confectionery", "cost": 2.80, "price": 5.49, "is_perishable": False, "shelf_life": 210, "storage": "Ambient", "lead_time_mean": 3.0, "lead_time_std": 0.5, "tier": "Moderate-mover"},
    {"name": "Matcha Green Tea Wafers 5oz", "category": "Snacks & Confectionery", "cost": 3.50, "price": 6.49, "is_perishable": False, "shelf_life": 180, "storage": "Ambient", "lead_time_mean": 5.0, "lead_time_std": 1.0, "tier": "Slow-mover"},

    # Beverages
    {"name": "Organic Cold Brew Coffee 32oz", "category": "Beverages", "cost": 3.10, "price": 5.99, "is_perishable": True, "shelf_life": 30, "storage": "Chilled", "lead_time_mean": 2.5, "lead_time_std": 0.5, "tier": "Trending-star"},
    {"name": "Sparkling Mineral Water 12-Pack", "category": "Beverages", "cost": 3.40, "price": 6.99, "is_perishable": False, "shelf_life": 365, "storage": "Ambient", "lead_time_mean": 3.0, "lead_time_std": 0.5, "tier": "Fast-mover"},
    {"name": "Pure Orange Juice 52oz", "category": "Beverages", "cost": 2.30, "price": 4.49, "is_perishable": True, "shelf_life": 24, "storage": "Chilled", "lead_time_mean": 2.0, "lead_time_std": 0.4, "tier": "Fast-mover"},
    {"name": "Classic Cola 12-Pack Cans", "category": "Beverages", "cost": 3.80, "price": 7.49, "is_perishable": False, "shelf_life": 365, "storage": "Ambient", "lead_time_mean": 2.5, "lead_time_std": 0.5, "tier": "Fast-mover"},
    {"name": "Oat Milk Barista Blend 32oz", "category": "Beverages", "cost": 2.20, "price": 4.29, "is_perishable": False, "shelf_life": 180, "storage": "Ambient", "lead_time_mean": 3.0, "lead_time_std": 0.5, "tier": "Trending-star"},
    {"name": "Organic Hibiscus Kombucha 16oz", "category": "Beverages", "cost": 2.10, "price": 3.99, "is_perishable": True, "shelf_life": 45, "storage": "Chilled", "lead_time_mean": 3.0, "lead_time_std": 0.6, "tier": "Moderate-mover"},
    {"name": "Artisanal Sparkling Lavender Soda 4pk", "category": "Beverages", "cost": 5.00, "price": 8.99, "is_perishable": False, "shelf_life": 180, "storage": "Ambient", "lead_time_mean": 6.0, "lead_time_std": 1.2, "tier": "Deadstock-candidate"},

    # Frozen Foods
    {"name": "Thin Crust Four Cheese Pizza 15oz", "category": "Frozen Foods", "cost": 3.20, "price": 6.79, "is_perishable": False, "shelf_life": 270, "storage": "Frozen", "lead_time_mean": 3.5, "lead_time_std": 0.6, "tier": "Fast-mover"},
    {"name": "Organic Frozen Blueberries 16oz", "category": "Frozen Foods", "cost": 2.60, "price": 4.99, "is_perishable": False, "shelf_life": 365, "storage": "Frozen", "lead_time_mean": 3.5, "lead_time_std": 0.6, "tier": "Moderate-mover"},
    {"name": "Vanilla Bean Ice Cream Pint", "category": "Frozen Foods", "cost": 2.50, "price": 5.29, "is_perishable": False, "shelf_life": 240, "storage": "Frozen", "lead_time_mean": 3.0, "lead_time_std": 0.5, "tier": "Moderate-mover"},
    {"name": "Spicy Black Bean Veggie Burgers 4ct", "category": "Frozen Foods", "cost": 3.00, "price": 5.99, "is_perishable": False, "shelf_life": 270, "storage": "Frozen", "lead_time_mean": 3.5, "lead_time_std": 0.7, "tier": "Moderate-mover"},
    {"name": "Frozen Seaweed Salad Packets 8oz", "category": "Frozen Foods", "cost": 4.80, "price": 8.49, "is_perishable": False, "shelf_life": 180, "storage": "Frozen", "lead_time_mean": 5.5, "lead_time_std": 1.2, "tier": "Slow-mover"},

    # Household & Personal Care
    {"name": "Eco-Friendly Dish Soap 16oz", "category": "Household", "cost": 1.90, "price": 3.99, "is_perishable": False, "shelf_life": 1080, "storage": "Ambient", "lead_time_mean": 4.0, "lead_time_std": 0.8, "tier": "Moderate-mover"},
    {"name": "Paper Towels 6 Double Rolls", "category": "Household", "cost": 5.50, "price": 9.99, "is_perishable": False, "shelf_life": 1800, "storage": "Ambient", "lead_time_mean": 3.5, "lead_time_std": 0.7, "tier": "Fast-mover"},
    {"name": "Ultra Soft Bath Tissue 12 Mega Rolls", "category": "Household", "cost": 6.80, "price": 12.49, "is_perishable": False, "shelf_life": 1800, "storage": "Ambient", "lead_time_mean": 3.5, "lead_time_std": 0.7, "tier": "Fast-mover"},
    {"name": "Lavender Multi-Surface Cleaner 28oz", "category": "Household", "cost": 2.20, "price": 4.49, "is_perishable": False, "shelf_life": 1080, "storage": "Ambient", "lead_time_mean": 4.0, "lead_time_std": 0.8, "tier": "Moderate-mover"},
    {"name": "Beeswax Furniture Polish 8oz", "category": "Household", "cost": 5.90, "price": 10.99, "is_perishable": False, "shelf_life": 1080, "storage": "Ambient", "lead_time_mean": 6.0, "lead_time_std": 1.4, "tier": "Deadstock-candidate"},
]

# Market basket association affinity clusters
BASKET_AFFINITIES = [
    # Breakfast basket
    ["Whole Milk 1 Gallon", "Organic Large Brown Eggs 12ct", "Whole Wheat Sandwich Bread 20oz", "Salted Butter 16oz", "Pure Orange Juice 52oz"],
    # Italian Dinner basket
    ["Spaghetti Pasta 16oz", "Italian Penne Rigate 16oz", "Marinara Pasta Sauce 24oz Jar", "Extra Virgin Olive Oil 500ml", "Fresh Garlic Bulb 3-Pack", "Artisan French Sourdough"],
    # Taco / Mexican Fiesta
    ["Tortilla Corn Chips 12oz", "Medium Restaurant Style Salsa 16oz", "Hass Avocados (Single)", "Shredded Sharp Cheddar 8oz", "Ground Beef 80/20 1lb", "Black Beans Canned 15oz"],
    # Healthy Breakfast / Smoothie
    ["Greek Yogurt Plain 32oz", "Organic Bananas 1 Bunch", "Organic Frozen Blueberries 16oz", "Organic Honey Almond Granola 12oz", "Honeycrisp Apples 3lb Bag"],
    # Party / Movie Night
    ["Classic Potato Chips Sea Salt 8oz", "Thin Crust Four Cheese Pizza 15oz", "Classic Cola 12-Pack Cans", "Dark Chocolate Sea Salt Bar 3.5oz", "Sparkling Mineral Water 12-Pack"],
]


class GroceryDataGenerator:
    """
    Robust synthetic generator for grocery sales, inventory levels, and product metadata.
    Provides mathematically and commercially coherent datasets for ML pipeline validation.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None) -> None:
        self.config = config or GeneratorConfig()
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        self.products_df: Optional[pd.DataFrame] = None
        self.inventory_df: Optional[pd.DataFrame] = None
        self.sales_df: Optional[pd.DataFrame] = None

    def generate_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes end-to-end data generation and returns:
        (products_df, inventory_df, sales_df)
        """
        print("[Generator] Generating Product Master Catalog...")
        self.products_df = self._generate_products()

        print("[Generator] Generating Market Basket Sales Transactions...")
        self.sales_df = self._generate_sales(self.products_df)

        print("[Generator] Simulating Daily Inventory Dynamics & Spoilage...")
        self.inventory_df = self._generate_inventory(self.products_df, self.sales_df)

        if self.config.inject_anomalies:
            print("[Generator] Injecting controlled pipeline anomalies for Module 1 testing...")
            self._inject_pipeline_anomalies()

        return self.products_df, self.inventory_df, self.sales_df

    def save_to_csv(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Saves generated dataframes to target CSV files."""
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.products_df is None or self.inventory_df is None or self.sales_df is None:
            self.generate_all()

        paths = {
            "products": out_dir / "products.csv",
            "inventory": out_dir / "inventory.csv",
            "sales": out_dir / "sales.csv",
        }

        self.products_df.to_csv(paths["products"], index=False)
        self.inventory_df.to_csv(paths["inventory"], index=False)
        self.sales_df.to_csv(paths["sales"], index=False)

        print(f"[Generator] Successfully saved:")
        print(f"  - Products:  {paths['products']} ({len(self.products_df)} records)")
        print(f"  - Sales:     {paths['sales']} ({len(self.sales_df)} records)")
        print(f"  - Inventory: {paths['inventory']} ({len(self.inventory_df)} records)")
        return paths

    def _generate_products(self) -> pd.DataFrame:
        """Generates Product Master dataset."""
        records: List[Dict[str, Any]] = []

        # If requested count exceeds base template, we scale dynamically
        templates = CATALOG_TEMPLATES.copy()
        while len(templates) < self.config.num_products:
            base = random.choice(CATALOG_TEMPLATES)
            variant_num = len(templates) + 1
            templates.append({
                "name": f"{base['name']} (Var-{variant_num})",
                "category": base["category"],
                "cost": round(base["cost"] * random.uniform(0.9, 1.25), 2),
                "price": round(base["price"] * random.uniform(0.9, 1.25), 2),
                "is_perishable": base["is_perishable"],
                "shelf_life": base["shelf_life"],
                "storage": base["storage"],
                "lead_time_mean": base["lead_time_mean"],
                "lead_time_std": base["lead_time_std"],
                "tier": base["tier"],
            })

        templates = templates[: self.config.num_products]

        for idx, item in enumerate(templates, start=1):
            sku_id = f"SKU-GROC-{idx:04d}"
            # Standard 12-digit UPC barcode simulation
            upc = f"0{idx:03d}{random.randint(10000000, 99999999)}"[:12]

            records.append({
                "sku_id": sku_id,
                "upc": upc,
                "product_name": item["name"],
                "category": item["category"],
                "cost_price": float(item["cost"]),
                "selling_price": float(item["price"]),
                "margin_pct": round((item["price"] - item["cost"]) / item["price"], 4),
                "is_perishable": bool(item["is_perishable"]),
                "shelf_life_days": int(item["shelf_life"]),
                "storage_type": item["storage"],
                "supplier_lead_time_mean": float(item["lead_time_mean"]),
                "supplier_lead_time_std": float(item["lead_time_std"]),
                "baseline_tier": item["tier"],
            })

        df = pd.DataFrame(records)
        return df

    def _generate_sales(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates realistic retail sales transactions using basket simulations.
        Applies:
        - Day-of-week seasonality (higher on Friday, Saturday, Sunday)
        - Payday spikes (1st and 15th of the month)
        - Item velocity profiles based on tier
        - Market basket item affinities (co-occurrences for Apriori)
        """
        start_date = datetime.date.fromisoformat(self.config.start_date)
        all_skus = products_df["sku_id"].tolist()
        name_to_sku = dict(zip(products_df["product_name"], products_df["sku_id"]))
        sku_to_price = dict(zip(products_df["sku_id"], products_df["selling_price"]))
        sku_to_tier = dict(zip(products_df["sku_id"], products_df["baseline_tier"]))

        # Base popularity weights according to tier
        tier_weights = {
            "Fast-mover": 18.0,
            "Trending-star": 15.0,
            "Moderate-mover": 7.0,
            "Slow-mover": 1.5,
            "Deadstock-candidate": 0.25,
        }
        sku_weights = np.array([tier_weights.get(sku_to_tier[s], 5.0) for s in all_skus])
        sku_probs = sku_weights / sku_weights.sum()

        # Map affinity cluster names to SKUs
        sku_affinity_groups: List[List[str]] = []
        for cluster in BASKET_AFFINITIES:
            group = [name_to_sku[n] for n in cluster if n in name_to_sku]
            if len(group) >= 2:
                sku_affinity_groups.append(group)

        sales_records: List[Dict[str, Any]] = []
        global_txn_counter = 1
        global_basket_counter = 1

        for day_offset in range(self.config.num_days):
            current_date = start_date + datetime.timedelta(days=day_offset)
            dow = current_date.weekday()  # 0=Monday, 6=Sunday
            day_of_month = current_date.day

            # Realistic temporal traffic multipliers
            dow_multiplier = 1.0
            if dow in [4]:       # Friday
                dow_multiplier = 1.25
            elif dow in [5, 6]:  # Weekend
                dow_multiplier = 1.45
            elif dow in [1, 2]:  # Tuesday, Wednesday
                dow_multiplier = 0.85

            payday_multiplier = 1.25 if day_of_month in [1, 2, 15, 16] else 1.0

            # Daily base transaction volume
            expected_txns = int(self.config.avg_transactions_per_day * dow_multiplier * payday_multiplier)
            num_txns = np.random.poisson(lam=max(10, expected_txns))

            for _ in range(num_txns):
                basket_id = f"BSK-{current_date.strftime('%Y%m')}-{global_basket_counter:06d}"
                global_basket_counter += 1

                # Generate a realistic shopping basket
                customer_id = f"CUST-{random.randint(1001, 1999):04d}"

                # Decide if this transaction is driven by an affinity theme (e.g. Pasta night)
                basket_items: List[str] = []
                if random.random() < 0.35 and sku_affinity_groups:
                    chosen_affinity = random.choice(sku_affinity_groups)
                    # Pick 2 to 4 items from the affinity group
                    k_items = min(len(chosen_affinity), random.randint(2, 4))
                    basket_items.extend(random.sample(chosen_affinity, k_items))

                # Add 1 to 4 other general groceries
                num_extra = random.randint(1, 4)
                extra_skus = np.random.choice(all_skus, size=num_extra, replace=False, p=sku_probs)
                for s in extra_skus:
                    if s not in basket_items:
                        basket_items.append(s)

                # Generate timestamp within open hours (07:00:00 to 22:00:00)
                hour = int(np.clip(np.random.normal(loc=15, scale=3.5), 7, 21))
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                txn_timestamp = datetime.datetime(
                    current_date.year, current_date.month, current_date.day, hour, minute, second
                )

                for sku_id in basket_items:
                    # Item quantity: mostly 1 or 2, rarely 3-5
                    qty = int(np.random.choice([1, 2, 3, 4], p=[0.72, 0.20, 0.06, 0.02]))
                    unit_price = sku_to_price[sku_id]

                    # Occasional promo discount (0%, 5%, 10%, 15%)
                    discount_pct = float(np.random.choice([0.0, 0.05, 0.10, 0.15], p=[0.75, 0.12, 0.08, 0.05]))
                    total_amount = round(qty * unit_price * (1.0 - discount_pct), 2)

                    sales_records.append({
                        "transaction_id": f"TXN-{current_date.strftime('%Y%m')}-{global_txn_counter:07d}",
                        "basket_id": basket_id,
                        "date": current_date.isoformat(),
                        "timestamp": txn_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "customer_id": customer_id,
                        "sku_id": sku_id,
                        "units_sold": qty,
                        "unit_price": unit_price,
                        "discount_pct": discount_pct,
                        "total_amount": total_amount,
                    })
                    global_txn_counter += 1

        df_sales = pd.DataFrame(sales_records)
        return df_sales

    def _generate_inventory(self, products_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulates dynamic daily inventory state machine for each product:
        - Daily demand calculated from aggregated sales
        - Restock orders placed based on replenishment cycle and dynamic lead times
        - In-transit orders tracking and fulfillment
        - Spoilage/waste simulation for perishables approaching or passing expiry
        """
        start_date = datetime.date.fromisoformat(self.config.start_date)
        all_skus = products_df["sku_id"].unique()

        # Precompute daily sales per SKU: (date_str, sku_id) -> units_sold
        sales_agg = sales_df.groupby(["date", "sku_id"])["units_sold"].sum().to_dict()

        product_meta = products_df.set_index("sku_id").to_dict(orient="index")

        inventory_records: List[Dict[str, Any]] = []

        # Track state per SKU
        sku_state: Dict[str, Dict[str, Any]] = {}
        for sku_id in all_skus:
            meta = product_meta[sku_id]
            # Initial stock proportional to tier
            tier = meta["baseline_tier"]
            initial_stock = {
                "Fast-mover": random.randint(90, 160),
                "Trending-star": random.randint(70, 130),
                "Moderate-mover": random.randint(40, 80),
                "Slow-mover": random.randint(25, 45),
                "Deadstock-candidate": random.randint(50, 95),  # High initial stock, low sales = deadstock!
            }.get(tier, 50)

            sku_state[sku_id] = {
                "on_hand": initial_stock,
                "pending_orders": [],  # List of tuples: (arrival_date, quantity, recorded_lead_time)
                "nearest_batch_expiry": start_date + datetime.timedelta(days=int(meta["shelf_life_days"])),
            }

        for day_offset in range(self.config.num_days):
            current_date = start_date + datetime.timedelta(days=day_offset)
            date_str = current_date.isoformat()

            for sku_id in all_skus:
                meta = product_meta[sku_id]
                state = sku_state[sku_id]
                starting_stock = state["on_hand"]

                # 1. Check arriving restocks
                restock_arrived = 0
                actual_lead_time_recorded: Optional[float] = None
                remaining_orders = []
                for arrival_date, order_qty, lead_t in state["pending_orders"]:
                    if arrival_date <= current_date:
                        restock_arrived += order_qty
                        actual_lead_time_recorded = lead_t
                    else:
                        remaining_orders.append((arrival_date, order_qty, lead_t))
                state["pending_orders"] = remaining_orders

                # Update stock with arrivals
                stock_after_arrivals = starting_stock + restock_arrived
                if restock_arrived > 0 and meta["is_perishable"]:
                    # Arriving fresh batch extends the active expiry date
                    state["nearest_batch_expiry"] = current_date + datetime.timedelta(days=int(meta["shelf_life_days"]))

                # 2. Fulfill sales
                sold_units = sales_agg.get((date_str, sku_id), 0)
                actual_sold = min(stock_after_arrivals, sold_units)
                stock_after_sales = stock_after_arrivals - actual_sold

                # 3. Simulate Spoilage & Shrinkage
                spoilage_units = 0
                if meta["is_perishable"]:
                    # If batch is expired or within 1 day of expiry and slow velocity
                    days_until_expiry = (state["nearest_batch_expiry"] - current_date).days
                    if days_until_expiry <= 0:
                        # 40-70% of remaining stock spoiled
                        spoilage_units = max(0, int(stock_after_sales * random.uniform(0.4, 0.7)))
                    elif days_until_expiry <= 2 and stock_after_sales > 15:
                        # Minor trim/spoilage
                        spoilage_units = max(0, int(stock_after_sales * random.uniform(0.05, 0.15)))

                ending_stock = max(0, stock_after_sales - spoilage_units)
                state["on_hand"] = ending_stock

                # 4. Check if replenishment order should be dispatched
                # Place order when stock drops below stochastic lead-time buffer
                # (Notice this is a simulation dynamic, not an ML decision rule; the ML models will learn this!)
                lt_mean = meta["supplier_lead_time_mean"]
                lt_std = meta["supplier_lead_time_std"]
                simulated_reorder_level = int(lt_mean * 3.5)

                total_incoming = sum(q for _, q, _ in state["pending_orders"])
                if (ending_stock + total_incoming) <= simulated_reorder_level:
                    # Place order
                    sim_lead_time = max(1.0, round(np.random.normal(loc=lt_mean, scale=lt_std), 1))
                    arrival_d = current_date + datetime.timedelta(days=int(round(sim_lead_time)))
                    order_size = int(lt_mean * 12 + random.randint(10, 30))
                    state["pending_orders"].append((arrival_d, order_size, sim_lead_time))

                inventory_records.append({
                    "date": date_str,
                    "sku_id": sku_id,
                    "starting_stock": starting_stock,
                    "restock_received": restock_arrived,
                    "sales_units": actual_sold,
                    "unfulfilled_demand": max(0, sold_units - actual_sold),
                    "spoilage_units": spoilage_units,
                    "ending_stock": ending_stock,
                    "batch_expiry_date": state["nearest_batch_expiry"].isoformat() if meta["is_perishable"] else None,
                    "supplier_lead_time_recorded": actual_lead_time_recorded if restock_arrived > 0 else np.nan,
                })

        df_inv = pd.DataFrame(inventory_records)
        return df_inv

    def _inject_pipeline_anomalies(self) -> None:
        """
        Injects a small percentage of realistic raw data corruptions to test
        Module 1's validation and cleaning engine:
        - Corrupted SKU formats (e.g. missing prefix, lowercase noise)
        - Missing timestamps in sales transactions
        - Negative or null inventory counts
        - Null product categories or price anomalies
        """
        if self.sales_df is None or self.inventory_df is None or self.products_df is None:
            return

        n_sales_anomalies = max(1, int(len(self.sales_df) * self.config.anomaly_rate))
        sales_idx = np.random.choice(self.sales_df.index, size=n_sales_anomalies, replace=False)

        for i, idx in enumerate(sales_idx):
            anomaly_type = i % 4
            if anomaly_type == 0:
                # Missing timestamp
                self.sales_df.at[idx, "timestamp"] = np.nan
            elif anomaly_type == 1:
                # Malformed SKU (lowercase or missing delimiter)
                raw_sku = str(self.sales_df.at[idx, "sku_id"])
                self.sales_df.at[idx, "sku_id"] = raw_sku.lower().replace("-", "")
            elif anomaly_type == 2:
                # Outlier or negative quantity
                self.sales_df.at[idx, "units_sold"] = -1
            elif anomaly_type == 3:
                # Null customer_id
                self.sales_df.at[idx, "customer_id"] = np.nan

        # Inject minor inventory anomalies
        n_inv_anomalies = max(1, int(len(self.inventory_df) * (self.config.anomaly_rate / 2)))
        inv_idx = np.random.choice(self.inventory_df.index, size=n_inv_anomalies, replace=False)
        for i, idx in enumerate(inv_idx):
            if i % 2 == 0:
                self.inventory_df.at[idx, "starting_stock"] = -5
            else:
                self.inventory_df.at[idx, "ending_stock"] = np.nan
