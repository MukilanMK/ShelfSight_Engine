# Module 6 (Unit-V) Ensemble Model Justification Report

## 1. Demand Velocity Regression Benchmark
| ensemble_model          | best_params                                                                   |   test_rmse |   test_mae |   test_r2 |   cv_rmse_mean |   cv_rmse_std |   overfitting_gap | is_overfitting   |
|:------------------------|:------------------------------------------------------------------------------|------------:|-----------:|----------:|---------------:|--------------:|------------------:|:-----------------|
| XGBoost                 | {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 120, 'subsample': 0.8} |      0.6783 |     0.3246 |    0.9961 |         0.6064 |        0.0185 |            0.235  | False            |
| Gradient Boosting       | {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}                   |      0.7051 |     0.3632 |    0.9958 |         0.6457 |        0.0136 |            0.2372 | False            |
| Random Forest (Bagging) | {'max_depth': None, 'n_estimators': 50}                                       |      0.818  |     0.2939 |    0.9943 |         0.8492 |        0.0399 |            0.5148 | True             |
| AdaBoost                | {'learning_rate': 0.1, 'n_estimators': 80}                                    |      2.2197 |     1.649  |    0.9583 |         2.1723 |        0.0402 |            0.0303 | False            |

**Selected Production Model**: **XGBoost**
- **Justification**: Exhibits optimal bias-variance tradeoff with minimal holdout RMSE and consistent cross-validation stability without overfitting.

## 2. Restock Urgency Classification Benchmark
| ensemble_model          | best_params                                                                  |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_ovr |   cv_f1_mean |   cv_f1_std |   overfitting_gap | is_overfitting   |
|:------------------------|:-----------------------------------------------------------------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-------------:|------------:|------------------:|:-----------------|
| AdaBoost                | {'learning_rate': 0.1, 'n_estimators': 40}                                   |     1      |            1      |         1      |     1      |             1 |       0.9997 |      0.0006 |            0.0003 | False            |
| Gradient Boosting       | {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 50}                  |     1      |            1      |         1      |     1      |             1 |       0.9997 |      0.0006 |            0.0003 | False            |
| Random Forest (Bagging) | {'criterion': 'gini', 'max_depth': 10, 'n_estimators': 100}                  |     0.9995 |            0.9998 |         0.9988 |     0.9993 |             1 |       0.9978 |      0.0017 |            0.0022 | False            |
| XGBoost                 | {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 60, 'subsample': 1.0} |     0.9963 |            0.9909 |         0.9923 |     0.9916 |             1 |       0.9919 |      0.0048 |            0.0081 | False            |

**Selected Production Model**: **AdaBoost**
- **Justification**: Achieves superior Macro F1 and ROC-AUC scores across multi-class restocking boundaries, demonstrating robust generalization over single-tree baselines.