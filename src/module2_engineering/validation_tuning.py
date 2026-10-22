"""
Module 2 (UNIT-I): Model Validation & Hyperparameter Tuning Engine
==================================================================
Implements:
1. Rigorous K-Fold Cross-Validation with train vs val score comparison
   to explicitly diagnose Overfitting vs Underfitting.
2. Automated Hyperparameter Tuning using GridSearchCV & RandomizedSearchCV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_validate,
)


@dataclass
class ValidationResults:
    """Stores detailed K-Fold cross validation evaluation metrics."""
    model_name: str
    train_scores_mean: float
    train_scores_std: float
    val_scores_mean: float
    val_scores_std: float
    overfitting_gap: float  # train_score - val_score
    is_overfitting: bool
    is_underfitting: bool
    fit_time_mean: float


@dataclass
class TuningResult:
    """Encapsulates hyperparameter search results."""
    model_name: str
    best_estimator: BaseEstimator
    best_params: Dict[str, Any]
    best_score: float
    cv_results_df: pd.DataFrame


class ModelTuner:
    """
    Standardized validation and hyperparameter optimization framework
    designed to eliminate arbitrary parameter choices.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def validate_with_kfold(
        self,
        model: BaseEstimator,
        model_name: str,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: Any,
        scoring: str = "r2",
    ) -> ValidationResults:
        """
        Executes K-Fold cross-validation tracking both training and validation
        loss to diagnose overfitting and underfitting.
        """
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        train_mean = float(np.mean(scores["train_score"]))
        train_std = float(np.std(scores["train_score"]))
        val_mean = float(np.mean(scores["test_score"]))
        val_std = float(np.std(scores["test_score"]))
        fit_time = float(np.mean(scores["fit_time"]))

        gap = train_mean - val_mean

        # Dynamic diagnostic criteria:
        # Overfitting if training score is substantially higher than validation score
        is_overfitting = (train_mean > 0.85 and gap > 0.15) or (gap > 0.25)
        # Underfitting if both train and validation scores are mediocre/poor
        is_underfitting = (train_mean < 0.40) and (val_mean < 0.40)

        return ValidationResults(
            model_name=model_name,
            train_scores_mean=round(train_mean, 4),
            train_scores_std=round(train_std, 4),
            val_scores_mean=round(val_mean, 4),
            val_scores_std=round(val_std, 4),
            overfitting_gap=round(gap, 4),
            is_overfitting=is_overfitting,
            is_underfitting=is_underfitting,
            fit_time_mean=round(fit_time, 4),
        )

    def tune_hyperparameters(
        self,
        model: BaseEstimator,
        model_name: str,
        param_grid: Dict[str, Any],
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: Any,
        search_type: str = "grid",
        n_iter: int = 15,
        scoring: str = "r2",
    ) -> TuningResult:
        """
        Tunes model hyperparameters across CV folds using Grid or Randomized search.
        """
        if search_type == "grid":
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                refit=True,
            )
        else:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=self.random_state,
                n_jobs=-1,
                refit=True,
            )

        search.fit(X, y)

        cv_df = pd.DataFrame(search.cv_results_)

        return TuningResult(
            model_name=model_name,
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            best_score=round(float(search.best_score_), 4),
            cv_results_df=cv_df,
        )
