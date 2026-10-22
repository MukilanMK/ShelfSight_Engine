Role: You are a Principal Machine Learning Engineer, Senior Data Architect, and Expert Data Science Instructor specializing in retail supply chain optimization and academic machine learning implementations.

Objective: Build a complete, production-grade, fully generalized, and syllabus-aligned Python and SQL-based Machine Learning system for a grocery store. The system must process raw sales, inventory, and product master data tables to dynamically identify trending items, calculate intelligent restocking needs, and detect deadstock/spoilage risks while strictly fulfilling all core machine learning curriculum units.

CRITICAL ARCHITECTURAL CONSTRAINTS - ZERO HARDCODING:
- Under no circumstances shall you use hardcoded thresholds, static magic numbers, or arbitrary rule-based cutoffs (e.g., DO NOT write `if days_without_sales > 60: deadstock = True` or `reorder_point = 10`).
- Every threshold, safety stock calculation, trend multiplier, classification boundary, and cluster cutoff must be dynamically computed or learned using statistical distributions, machine learning optimization, or data-driven modeling based on the live dataset.

The system must be structured into modular components mapped directly to the following technical and academic requirements:

---

### MODULE 1: Data Ingestion, Validation & Feature Engineering
- Invalidate malformed records, handle missing timestamps, and standardize product IDs (SKU/UPC).
- Automatically parse temporal features (day of week, month, seasonality index, holidays, paydays).
- Compute dynamic rolling velocity metrics (7-day, 14-day, 30-day moving averages), dynamic supplier lead times, Days of Inventory Remaining (DIR), and Gross Margin Return on Investment (GMROI).

---

### MODULE 2 (UNIT-I): ML Basics & Model Engineering
- Process data through rigorous preprocessing, split pipelines, and dynamic feature scaling.
- Implement Model Validation Techniques: Perform K-Fold Cross-Validation on all predictive models to prevent Overfitting and Underfitting.
- Apply Hyperparameter Tuning (using GridSearchCV or RandomizedSearchCV) across training runs.

---

### MODULE 3 (UNIT-II): Feature Selection Techniques
Before model training, apply multi-method feature selection on the engineered grocery metrics:
- Filter Methods: Implement Univariate Selection (SelectKBest) and Random Forest Feature Importance.
- Wrapper Methods: Implement Recursive Feature Elimination (RFE) or Forward/Backward selection.
- Embedded Methods: Apply LASSO Regularization (L1 penalty) to isolate top inventory predictors and drop noise.

---

### MODULE 4 (UNIT-III): Supervised Machine Learning
- Demand & Sales Forecasting (Regression):
  * Train and compare Linear Regression, Ridge Regression, and Lasso Regression to predict daily item sales velocity.
- Restock & Inventory Action Classifier (Classification):
  * Implement and compare Logistic Regression, Decision Tree Classifier (ID3/CART philosophy), Naïve Bayes, Support Vector Machines (SVM), and K-Nearest Neighbors (K-NN) to classify items into dynamic restocking urgency tiers.

---

### MODULE 5 (UNIT-IV): Unsupervised Machine Learning (Segmentation & Deadstock)
- Segmentation & Cluster Analysis:
  * Apply K-Means Clustering and Density-Based Clustering (DBSCAN) to group products into dynamic performance health tiers (Fast-movers, Slow-movers, At-Risk/Deadstock).
  * Implement Agglomerative Hierarchical Clustering to evaluate product similarity trees based on sales trends and margins.
  * Evaluate clusters to automatically highlight high-risk deadstock and spoilage risks without static age cutoffs.
  * Implement Association Rule Mining (Apriori algorithm) to find co-purchase patterns dynamically for cross-selling and bundling slow movers.

---

### MODULE 6 (UNIT-V): Ensemble Techniques (Model Optimization & Boosting)
- Train advanced ensemble models to optimize sales trend prediction and out-of-stock risk scoring:
  * Bagging & Random Forest: Random Forest Regressor/Classifier with hyperparameter tuning.
  * Boosting: Train and tune Gradient Boosting, AdaBoost, and XGBoost models.
- Output comparative performance metrics (RMSE/MAE for regression; Accuracy, F1-Score, ROC-AUC for classification) to justify the best final model choice.

---

### OUTPUT & DELIVERABLES:
1. Synthetic/sample dataset generator for grocery sales, inventory levels, and product metadata.
2. Complete, modular, type-hinted Python code split logically by syllabus unit modules with robust error handling and unit tests.
3. Clean summary dashboard output displaying:
   - Top Trending SKUs (via ensemble model sales velocity predictions & Z-score trend acceleration).
   - Restock Action List (via classification & dynamic safety stock models).
   - Deadstock / Spoilage Risk Report with automated dynamic markdown percentage recommendations (via K-Means / Hierarchical Clustering).
4. A comprehensive documentation file explaining how the dynamic, non-hardcoded algorithms adapt to data shifts and fulfill the academic syllabus criteria.