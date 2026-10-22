"""
Unit Tests for Dashboard & Executive Reporting
==============================================
Verifies:
1. ExecutiveSummaryReporter compiles trending items, restock action list,
   deadstock report, and co-purchase bundles.
2. Markdown and HTML dashboard reports are generated without error.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.dashboard.summary_reporter import ExecutiveSummaryReporter
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR, MODELS_DIR


def test_executive_summary_reporter_generation():
    reporter = ExecutiveSummaryReporter(
        features_path=PROCESSED_DATA_DIR / "daily_product_features.csv",
        selected_features_path=PROCESSED_DATA_DIR / "selected_features.json",
        products_path=RAW_DATA_DIR / "products.csv",
        reports_dir=REPORTS_DIR,
        models_dir=MODELS_DIR,
    )
    results = reporter.generate_dashboard()

    assert "trending_skus" in results
    assert "restock_actions" in results
    assert "deadstock_spoilage" in results
    assert "bundles" in results

    assert len(results["trending_skus"]) > 0
    assert len(results["restock_actions"]) > 0

    md_path = REPORTS_DIR / "executive_summary_dashboard.md"
    html_path = REPORTS_DIR / "dashboard.html"
    assert md_path.exists()
    assert html_path.exists()
