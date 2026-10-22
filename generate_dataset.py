"""
Dataset Generation CLI Entrypoint
================================
Generates raw datasets according to project specification:
- products.csv
- inventory.csv
- sales.csv

Usage:
    python generate_dataset.py --num-products 60 --days 180 --output-dir data/raw
"""

import argparse
from pathlib import Path
from src.generator.synthetic_data_generator import GroceryDataGenerator, GeneratorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Grocery Data Generator")
    parser.add_argument("--num-products", type=int, default=60, help="Number of distinct products")
    parser.add_argument("--days", type=int, default=180, help="Number of simulated days")
    parser.add_argument("--avg-txns", type=int, default=140, help="Average transactions per day")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Simulation start date (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory for CSV files")
    parser.add_argument("--no-anomalies", action="store_true", help="Disable injection of pipeline validation anomalies")

    args = parser.parse_args()

    config = GeneratorConfig(
        num_products=args.num_products,
        start_date=args.start_date,
        num_days=args.days,
        avg_transactions_per_day=args.avg_txns,
        random_seed=args.seed,
        output_dir=Path(args.output_dir),
        inject_anomalies=not args.no_anomalies,
    )

    print("=====================================================================")
    print("      GROCERY STORE SYNTHETIC DATASET GENERATION PIPELINE           ")
    print("=====================================================================")
    print(f"Products Count:       {config.num_products}")
    print(f"Simulation Horizon:   {config.num_days} days (from {config.start_date})")
    print(f"Avg Daily Txns:       {config.avg_transactions_per_day}")
    print(f"Anomaly Injection:    {config.inject_anomalies}")
    print(f"Output Directory:     {config.output_dir.resolve()}")
    print("=====================================================================")

    generator = GroceryDataGenerator(config)
    paths = generator.save_to_csv()

    print("\n Generation complete! Data files are ready for Module 1 processing.")


if __name__ == "__main__":
    main()
