"""
Module 4 (UNIT-III): Supervised Machine Learning
"""
from src.module4_supervised.regression_demand import DemandForecaster, RegressionBenchmark
from src.module4_supervised.classification_restock import RestockClassifier, ClassificationBenchmark
from src.module4_supervised.pipeline import run_module4_pipeline

__all__ = [
    "DemandForecaster",
    "RegressionBenchmark",
    "RestockClassifier",
    "ClassificationBenchmark",
    "run_module4_pipeline",
]
