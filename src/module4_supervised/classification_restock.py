"""
Module 4 (UNIT-III): Supervised Restock Urgency Action Classifier
=================================================================
Trains, tunes, and systematically compares 5 core classification paradigms:
1. Logistic Regression (Multinomial with L2 regularization)
2. Decision Tree Classifier (CART & ID3 entropy/gini principles)
3. Gaussian Naïve Bayes (Probabilistic generative classifier)
4. Support Vector Classifier - SVM (RBF kernel & hyperplane margin maximization)
5. K-Nearest Neighbors - K-NN (Instance-based lazy learning)

Classifies inventory into dynamic restocking tiers:
- Tier 0: CRITICAL_RESTOCK
- Tier 1: REORDER_RECOMMENDED
- Tier 2: OPTIMAL_STOCK
- Tier 3: EXCESS_OVERSTOCK
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import MODELS_DIR, REPORTS_DIR
from src.module2_engineering.validation_tuning import ModelTuner, ValidationResults


@dataclass
class ClassificationModelMetrics:
    """Evaluation metrics for a single classifier."""
    model_name: str
    best_params: Dict[str, Any]
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc_ovr: Optional[float]
    cv_f1_mean: float
    cv_f1_std: float
    overfitting_gap: float
    is_overfitting: bool
    model_obj: BaseEstimator


@dataclass
class ClassificationBenchmark:
    """Benchmark comparative summary across all 5 classification models."""
    comparison_table: pd.DataFrame
    best_model_name: str
    best_model: BaseEstimator
    models: Dict[str, ClassificationModelMetrics]


class RestockClassifier:
    """
    Trains, validates, tunes, and compares the 5 syllabus classification algorithms.
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
    ) -> ClassificationBenchmark:
        """
        Executes model training and benchmark comparison across the 5 classifier architectures.
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        results: Dict[str, ClassificationModelMetrics] = {}

        # 1. Logistic Regression (Multinomial)
        print("[RestockClassifier] 1. Training Logistic Regression...")
        log_reg = LogisticRegression(max_iter=1000, random_state=self.random_state)
        log_cv = self.tuner.validate_with_kfold(
            model=log_reg,
            model_name="LogisticRegression",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="f1_macro",
        )
        log_reg.fit(X_train, y_train)
        results["LogisticRegression"] = self._evaluate_model(
            name="LogisticRegression",
            model=log_reg,
            params={"max_iter": 1000},
            X_test=X_test,
            y_test=y_test,
            cv_res=log_cv,
        )

        # 2. Decision Tree Classifier (CART / ID3 philosophy)
        print("[RestockClassifier] 2. Tuning Decision Tree Classifier (Entropy / Gini)...")
        dt_param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [4, 6, 8, 12, None],
            "min_samples_split": [2, 5, 10],
        }
        dt_tuned = self.tuner.tune_hyperparameters(
            model=DecisionTreeClassifier(random_state=self.random_state),
            model_name="DecisionTree",
            param_grid=dt_param_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring="f1_macro",
        )
        dt_model = dt_tuned.best_estimator
        dt_cv = self.tuner.validate_with_kfold(
            model=dt_model,
            model_name="DecisionTree",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="f1_macro",
        )
        results["DecisionTree"] = self._evaluate_model(
            name="DecisionTree",
            model=dt_model,
            params=dt_tuned.best_params,
            X_test=X_test,
            y_test=y_test,
            cv_res=dt_cv,
        )

        # 3. Gaussian Naïve Bayes
        print("[RestockClassifier] 3. Training Gaussian Naïve Bayes...")
        gnb = GaussianNB()
        gnb_cv = self.tuner.validate_with_kfold(
            model=gnb,
            model_name="NaiveBayes",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="f1_macro",
        )
        gnb.fit(X_train, y_train)
        results["NaiveBayes"] = self._evaluate_model(
            name="NaiveBayes",
            model=gnb,
            params={},
            X_test=X_test,
            y_test=y_test,
            cv_res=gnb_cv,
        )

        # 4. Support Vector Machine (SVM)
        print("[RestockClassifier] 4. Training Support Vector Classifier (SVM RBF Kernel)...")
        # Use CalibratedClassifierCV to provide well-calibrated class probabilities without scikit-learn deprecation
        try:
            from sklearn.calibration import CalibratedClassifierCV
            svm_model = CalibratedClassifierCV(estimator=SVC(kernel="rbf", C=2.0, random_state=self.random_state), ensemble=False)
        except Exception:
            svm_model = SVC(kernel="rbf", C=2.0, probability=True, random_state=self.random_state)
        svm_cv = self.tuner.validate_with_kfold(
            model=svm_model,
            model_name="SVM",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="f1_macro",
        )
        svm_model.fit(X_train, y_train)
        results["SVM"] = self._evaluate_model(
            name="SVM",
            model=svm_model,
            params={"kernel": "rbf", "C": 2.0},
            X_test=X_test,
            y_test=y_test,
            cv_res=svm_cv,
        )

        # 5. K-Nearest Neighbors (K-NN)
        print("[RestockClassifier] 5. Tuning K-Nearest Neighbors (K-NN)...")
        knn_grid = {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}
        knn_tuned = self.tuner.tune_hyperparameters(
            model=KNeighborsClassifier(),
            model_name="KNN",
            param_grid=knn_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring="f1_macro",
        )
        knn_model = knn_tuned.best_estimator
        knn_cv = self.tuner.validate_with_kfold(
            model=knn_model,
            model_name="KNN",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="f1_macro",
        )
        results["KNN"] = self._evaluate_model(
            name="KNN",
            model=knn_model,
            params=knn_tuned.best_params,
            X_test=X_test,
            y_test=y_test,
            cv_res=knn_cv,
        )

        # Build comparative leaderboard
        rows = []
        for name, m in results.items():
            rows.append({
                "model": name,
                "best_params": str(m.best_params),
                "accuracy": m.accuracy,
                "precision_macro": m.precision_macro,
                "recall_macro": m.recall_macro,
                "f1_macro": m.f1_macro,
                "roc_auc_ovr": m.roc_auc_ovr,
                "cv_f1_mean": m.cv_f1_mean,
                "cv_f1_std": m.cv_f1_std,
                "overfitting_gap": m.overfitting_gap,
                "is_overfitting": m.is_overfitting,
            })
        comparison_df = pd.DataFrame(rows).sort_values(by="f1_macro", ascending=True).reset_index(drop=True)
        # Invert to have highest F1 at the top
        comparison_df = comparison_df.iloc[::-1].reset_index(drop=True)

        best_model_name = comparison_df.iloc[0]["model"]
        best_model_obj = results[best_model_name].model_obj

        print(f"\n[RestockClassifier] Classification Benchmark Leaderboard:")
        print(comparison_df.to_string(index=False))
        print(f" Best Classifier: {best_model_name} (F1-Macro={comparison_df.iloc[0]['f1_macro']:.4f}, Accuracy={comparison_df.iloc[0]['accuracy']:.4f})")

        return ClassificationBenchmark(
            comparison_table=comparison_df,
            best_model_name=best_model_name,
            best_model=best_model_obj,
            models=results,
        )

    def _evaluate_model(
        self,
        name: str,
        model: BaseEstimator,
        params: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_res: ValidationResults,
    ) -> ClassificationModelMetrics:
        """Calculates multi-class classification metrics."""
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        roc_auc = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
                roc_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                roc_auc = None

        return ClassificationModelMetrics(
            model_name=name,
            best_params=params,
            accuracy=round(acc, 4),
            precision_macro=round(prec, 4),
            recall_macro=round(rec, 4),
            f1_macro=round(f1, 4),
            roc_auc_ovr=round(roc_auc, 4) if roc_auc is not None else None,
            cv_f1_mean=cv_res.val_scores_mean,
            cv_f1_std=cv_res.val_scores_std,
            overfitting_gap=cv_res.overfitting_gap,
            is_overfitting=cv_res.is_overfitting,
            model_obj=model,
        )

    def save_best_model(self, benchmark: ClassificationBenchmark, output_dir: Path = MODELS_DIR) -> Path:
        """Saves best classifier artifact to disk."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "best_restock_classifier.pkl"
        joblib.dump(benchmark.best_model, model_path)
        print(f"[RestockClassifier] Saved {benchmark.best_model_name} model to {model_path}")
        return model_path
