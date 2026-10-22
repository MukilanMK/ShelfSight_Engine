"""
Module 4 (UNIT-III): Supervised Demand & Sales Forecasting (Regression)
======================================================================
Trains, tunes, and systematically compares:
1. Ordinary Least Squares (OLS) Linear Regression
2. Ridge Regression (L2 Regularization)
3. Lasso Regression (L1 Sparsity Regularization)

Evaluates out-of-sample generalization with RMSE, MAE, and R^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.config import MODELS_DIR, REPORTS_DIR
from src.module2_engineering.validation_tuning import ModelTuner, ValidationResults


@dataclass
class RegressionModelMetrics:
    """Evaluation metrics for a single regression model."""
    model_name: str
    best_params: Dict[str, Any]
    test_rmse: float
    test_mae: float
    test_r2: float
    cv_val_r2_mean: float
    cv_val_r2_std: float
    overfitting_gap: float
    is_overfitting: bool
    model_obj: BaseEstimator


@dataclass
class RegressionBenchmark:
    """Collection of evaluated regression models and comparative metrics."""
    comparison_table: pd.DataFrame
    best_model_name: str
    best_model: BaseEstimator
    models: Dict[str, RegressionModelMetrics]


class DemandForecaster:
    """
    Trains, tunes, and evaluates linear regression models for daily sales velocity.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.tuner = ModelTuner(random_state=random_state)

    def train_and_compare(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_splits: int = 5,
    ) -> RegressionBenchmark:
        """
        Fits Linear Regression, Ridge, and Lasso, performing K-Fold CV and hyperparameter tuning.
        """
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        results: Dict[str, RegressionModelMetrics] = {}

        # 1. Linear Regression (Baseline OLS)
        print("[DemandForecaster] 1. Training Linear Regression (OLS)...")
        ols = LinearRegression()
        ols_cv = self.tuner.validate_with_kfold(
            model=ols,
            model_name="LinearRegression",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="r2",
        )
        ols.fit(X_train, y_train)
        y_pred_ols = ols.predict(X_test)
        results["LinearRegression"] = self._compute_metrics(
            name="LinearRegression",
            model=ols,
            params={},
            y_test=y_test,
            y_pred=y_pred_ols,
            cv_res=ols_cv,
        )

        # 2. Ridge Regression (L2) with Hyperparameter Tuning
        print("[DemandForecaster] 2. Tuning Ridge Regression (L2 Penalty)...")
        ridge_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]}
        ridge_tuned = self.tuner.tune_hyperparameters(
            model=Ridge(random_state=self.random_state),
            model_name="Ridge",
            param_grid=ridge_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring="r2",
        )
        ridge_model = ridge_tuned.best_estimator
        ridge_cv = self.tuner.validate_with_kfold(
            model=ridge_model,
            model_name="Ridge",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="r2",
        )
        y_pred_ridge = ridge_model.predict(X_test)
        results["Ridge"] = self._compute_metrics(
            name="Ridge",
            model=ridge_model,
            params=ridge_tuned.best_params,
            y_test=y_test,
            y_pred=y_pred_ridge,
            cv_res=ridge_cv,
        )

        # 3. Lasso Regression (L1) with Hyperparameter Tuning
        print("[DemandForecaster] 3. Tuning Lasso Regression (L1 Penalty)...")
        lasso_grid = {"alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]}
        lasso_tuned = self.tuner.tune_hyperparameters(
            model=Lasso(random_state=self.random_state, max_iter=3000),
            model_name="Lasso",
            param_grid=lasso_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring="r2",
        )
        lasso_model = lasso_tuned.best_estimator
        lasso_cv = self.tuner.validate_with_kfold(
            model=lasso_model,
            model_name="Lasso",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="r2",
        )
        y_pred_lasso = lasso_model.predict(X_test)
        results["Lasso"] = self._compute_metrics(
            name="Lasso",
            model=lasso_model,
            params=lasso_tuned.best_params,
            y_test=y_test,
            y_pred=y_pred_lasso,
            cv_res=lasso_cv,
        )

        # Assemble comparative leaderboard
        rows = []
        for name, m in results.items():
            rows.append({
                "model": name,
                "best_params": str(m.best_params),
                "test_rmse": m.test_rmse,
                "test_mae": m.test_mae,
                "test_r2": m.test_r2,
                "cv_r2_mean": m.cv_val_r2_mean,
                "cv_r2_std": m.cv_val_r2_std,
                "overfitting_gap": m.overfitting_gap,
                "is_overfitting": m.is_overfitting,
            })
        comparison_df = pd.DataFrame(rows).sort_values(by="test_rmse", ascending=True).reset_index(drop=True)

        best_model_name = comparison_df.iloc[0]["model"]
        best_model_obj = results[best_model_name].model_obj

        print(f"\n[DemandForecaster] Regression Benchmark Results:")
        print(comparison_df.to_string(index=False))
        print(f" Best Performer: {best_model_name} (RMSE={comparison_df.iloc[0]['test_rmse']:.4f}, R2={comparison_df.iloc[0]['test_r2']:.4f})")

        return RegressionBenchmark(
            comparison_table=comparison_df,
            best_model_name=best_model_name,
            best_model=best_model_obj,
            models=results,
        )

    def _compute_metrics(
        self,
        name: str,
        model: BaseEstimator,
        params: Dict[str, Any],
        y_test: pd.Series,
        y_pred: np.ndarray,
        cv_res: ValidationResults,
    ) -> RegressionModelMetrics:
        """Calculates regression metrics."""
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        return RegressionModelMetrics(
            model_name=name,
            best_params=params,
            test_rmse=round(rmse, 4),
            test_mae=round(mae, 4),
            test_r2=round(r2, 4),
            cv_val_r2_mean=cv_res.val_scores_mean,
            cv_val_r2_std=cv_res.val_scores_std,
            overfitting_gap=cv_res.overfitting_gap,
            is_overfitting=cv_res.is_overfitting,
            model_obj=model,
        )

    def save_best_model(self, benchmark: RegressionBenchmark, output_dir: Path = MODELS_DIR) -> Path:
        """Saves best regression model artifact to disk."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "best_demand_regressor.pkl"
        joblib.dump(benchmark.best_model, model_path)
        print(f"[DemandForecaster] Saved {benchmark.best_model_name} model to {model_path}")
        return model_path
