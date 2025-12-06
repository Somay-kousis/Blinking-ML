# Blinking-ML
Just getting started again 
---

**Total: 180 hours | 6 hours/day | Daily Math: 30 mins before ML work**

---

## DAILY MATH ROUTINE (30 mins/day)
- **Week 1:** [3Blue1Brown - Essence of Linear Algebra](https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (1 video/day)
- **Week 2:** [3Blue1Brown - Essence of Calculus](https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) (1 video/day)  
- **Week 3-4:** [StatQuest](https://youtube.com/c/joshstarmer) videos based on daily ML topic

---

## WEEK 1: LINEAR MODELS (45 hours)

### Day 1: Linear Regression Foundations
**Goal:** Understand and implement linear regression from scratch

**Watch (45m):**
- StatQuest: [Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo) (27m)
- StatQuest: [Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8) (18m)

**Code from Scratch (2.5h):**
- Generate synthetic data: `y = 4 + 3x + noise`
- Implement cost function (MSE)
- Implement gradient descent
- Train and visualize results
- Plot cost history
- **Reference if stuck:** https://github.com/SSaishruthi/Linear_Regression_from_Scratch

**Sklearn Tutorial (2h):**
- Work through: https://scikit-learn.org/stable/getting_started.html
- Work through: https://scikit-learn.org/stable/tutorial/basic/tutorial.html
- **DO EVERY CODE EXAMPLE**

**Practice (30m):**
- Recreate scratch implementation using `sklearn.linear_model.LinearRegression`
- Use `train_test_split`, calculate MSE and R²
- Compare results

**Deliverable:** `day1_linear_regression.ipynb` on GitHub

---

### Day 2: Multiple Regression & Vectorization
**Goal:** Master vectorized operations and multiple features

**Watch (30m):**
- StatQuest: [Multiple Regression](https://www.youtube.com/watch?v=zITIFTsivN8)

**Code from Scratch (2h):**
- Generate data with 5 features using `sklearn.datasets.make_regression`
- Implement vectorized gradient descent (no loops in training)
- Implement normal equation: `θ = (X^T X)^-1 X^T y`
- Compare both methods (speed and results)

**Real Dataset Practice (2h):**
- Load: `sklearn.datasets.fetch_california_housing`
- Train/test split (80-20)
- Apply both your methods
- Time comparison
- Error analysis

**Sklearn Deep Dive (1h):**
- Read: https://scikit-learn.org/stable/modules/linear_model.html
- Try: `LinearRegression` with different parameters
- Experiment with different test sizes

**Deliverable:** `day2_multiple_regression.ipynb`

---

### Day 3: Regularization (Ridge/Lasso)
**Goal:** Understand and implement L1/L2 regularization

**Watch (30m):**
- StatQuest: [Ridge Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) (20m)
- StatQuest: [Lasso Regression](https://www.youtube.com/watch?v=NGf0voTMlcs) (8m)

**Code from Scratch (2.5h):**
- Implement Ridge regression (L2 penalty in cost + gradients)
- Implement Lasso regression (L1 penalty, use soft thresholding)
- Create dataset with correlated features
- Visualize coefficient shrinkage

**Sklearn Regularization (1.5h):**
- Use `Ridge`, `Lasso`, `ElasticNet` on california_housing
- Try alphas: `[0.01, 0.1, 1, 10, 100]`
- Plot coefficient paths (alpha vs coefficients)
- Plot validation curves (alpha vs MSE)
- Compare all three models

**Read (1h):**
- https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression
- https://scikit-learn.org/stable/modules/linear_model.html#lasso

**Deliverable:** `day3_regularization.ipynb`

---

### Day 4: Logistic Regression
**Goal:** Binary classification and decision boundaries

**Watch (30m):**
- StatQuest: [Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) (9m)
- StatQuest: [Maximum Likelihood](https://www.youtube.com/watch?v=BfKanl1aSG0) (10m)

**Code from Scratch (3h):**
- Implement sigmoid function
- Implement binary cross-entropy loss
- Implement gradient descent for logistic regression
- Test on `sklearn.datasets.make_classification`
- Plot decision boundary (2D visualization)

**Sklearn Practice (1h):**
- Use `LogisticRegression` on `load_breast_cancer`
- Try different solvers: `['lbfgs', 'liblinear', 'saga']`
- Try different penalties: `['l1', 'l2', 'elasticnet']`
- Compare with your implementation

**Read (1h):**
- https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

**Deliverable:** `day4_logistic_regression.ipynb`

---

### Day 5: Classification Metrics
**Goal:** Master evaluation metrics for classification

**Watch (45m):**
- StatQuest: [Sensitivity and Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8) (12m)
- StatQuest: [ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM) (17m)
- StatQuest: [Precision and Recall](https://www.youtube.com/watch?v=jJ7ff7Gcq34) (6m)

**Code from Scratch (2h):**
- Implement confusion matrix
- Implement: accuracy, precision, recall, F1-score
- Implement ROC curve calculation (tricky - take time)
- Implement AUC calculation
- Test against `sklearn.metrics`

**Sklearn Metrics (1.5h):**
- Load breast_cancer dataset
- Train logistic regression
- Calculate ALL metrics
- Plot ROC curve with AUC
- Plot precision-recall curve
- Plot confusion matrix heatmap (seaborn)

**Read (1.5h):**
- https://scikit-learn.org/stable/modules/model_evaluation.html
- **Read entire Classification Metrics section**

**Deliverable:** `day5_classification_metrics.ipynb`

---

### Day 6: Sklearn Workflow Mastery
**Goal:** Pipelines, cross-validation, preprocessing

**Sklearn Official Tutorial (3h):**
- https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
- https://scikit-learn.org/stable/modules/cross_validation.html
- https://scikit-learn.org/stable/modules/preprocessing.html
- **DO EVERY CODE EXAMPLE IN JUPYTER**

**Code Practice (2.5h):**
Dataset: `load_diabetes` or `california_housing`

Implement:
1. Train-test split with stratification
2. K-fold cross-validation (5 folds)
3. `StandardScaler` vs `MinMaxScaler` comparison
4. Create Pipeline: `scaler → model`
5. `GridSearchCV` on pipeline with multiple parameters
6. `ColumnTransformer` for mixed data types
7. Save/load model with `joblib`

**Deliverable:** `day6_sklearn_workflow.ipynb`

---

### Day 7: PROJECT - House Prices Kaggle
**Goal:** End-to-end ML project with everything learned

**Competition:** [House Prices - Advanced Regression](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**EDA (2h):**
- Load train/test data
- Missing values analysis (heatmap)
- Numerical features: distributions, correlations
- Categorical features: count plots, relationship with target
- Outlier detection and handling
- Document insights

**Feature Engineering (1.5h):**
- Handle missing values (strategy for each column)
- Log transform skewed features
- Create new features (TotalSF, Age, etc.)
- Encode categorical variables
- Drop high correlation features
- Feature scaling

**Modeling (1.5h):**
- Try models: Linear, Ridge, Lasso, ElasticNet
- 5-fold cross-validation for each
- Plot: alpha vs RMSE for regularized models
- Select best model
- Hyperparameter tuning with GridSearchCV

**Submission (30m):**
- Predict on test set
- Create submission.csv
- Submit to Kaggle
- Screenshot score

**Deliverable:** `day7_house_prices.ipynb` + detailed README on GitHub

---

## WEEK 2: TREE-BASED MODELS (45 hours)

### Day 8: Decision Trees from Scratch
**Goal:** Understand tree building algorithms

**Watch (30m):**
- StatQuest: [Decision Trees](https://www.youtube.com/watch?v=_L39rN6gz7Y) (17m)
- StatQuest: [Decision Trees Part 2](https://www.youtube.com/watch?v=wpNl-JwwplA) (10m)

**Code from Scratch (3.5h):**
Implement ID3 algorithm:
- `Node` class (feature, threshold, left, right, value)
- `entropy(y)` function
- `information_gain(X, y, feature, threshold)` function
- `best_split(X, y)` - try all features and thresholds
- `build_tree(X, y, depth, max_depth)` - recursive
- `predict(x, node)` function
- Test on iris dataset
- **Reference if stuck:** https://github.com/SebastianMantey/Decision-Tree-from-Scratch

**Sklearn Trees (1.5h):**
- Use `DecisionTreeClassifier` on iris
- Visualize tree with `plot_tree`
- Try max_depths: `[2, 3, 5, 10, None]`
- Plot: depth vs accuracy

**Deliverable:** `day8_decision_trees.ipynb`

---

### Day 9: CART & Sklearn Deep Dive
**Goal:** Gini index and advanced tree concepts

**Watch (20m):**
- StatQuest: [Regression Trees](https://www.youtube.com/watch?v=g9c66TUylZ4) (15m)

**Code Gini Implementation (1.5h):**
- Implement `gini_index(y)` function
- Implement `gini_split(X, y, feature, threshold)`
- Modify Day 8 tree to use Gini instead of entropy
- Compare both on same dataset

**Sklearn Experiments (3.5h):**
Datasets: `load_breast_cancer` (classification), `load_diabetes` (regression)

Hyperparameter grid:
```python
{
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}
```
- Run GridSearchCV
- Analyze feature importances (bar plot)
- Visualize best tree
- Compare with default parameters

**Read (30m):**
- https://scikit-learn.org/stable/modules/tree.html

**Deliverable:** `day9_cart_trees.ipynb`

---

### Day 10: Random Forest from Scratch
**Goal:** Understand ensemble learning and bagging

**Watch (30m):**
- StatQuest: [Random Forests Part 1](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) (10m)
- StatQuest: [Random Forests Part 2](https://www.youtube.com/watch?v=nyxTdL_4Q-Q) (14m)

**Code from Scratch (3h):**
Build simple Random Forest:
- `bootstrap_sample(X, y)` - sampling with replacement
- `RandomForest` class with `n_trees`, `max_depth`, `max_features`
- `fit()` - train multiple trees with random feature subsets
- `predict()` - majority voting for classification
- Test on `make_classification`
- Compare: 1 tree vs 10 vs 100 trees

**Sklearn Random Forest (2h):**
- Use `RandomForestClassifier` on breast_cancer
- Try n_estimators: `[10, 50, 100, 200]`
- Plot: n_estimators vs accuracy
- Feature importance analysis
- Compare OOB score vs validation score

**Deliverable:** `day10_random_forest.ipynb`

---

### Day 11: Random Forest Mastery
**Goal:** Advanced RF tuning and applications

**Read (1h):**
- https://scikit-learn.org/stable/modules/ensemble.html#forest
- Read entire Random Forest section

**Hands-on Practice (4.5h):**
Dataset: `load_digits` (harder, 64 features)

Use `RandomizedSearchCV` (faster than Grid):
```python
{
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}
```
- 50 iterations, 5-fold CV (will take 30+ mins)
- Analyze results
- Feature importance: plot top 20
- Train with only top 50% features, compare
- Learning curves: plot training size vs score
- Compare RF vs single Decision Tree

**Deliverable:** `day11_rf_advanced.ipynb`

---

### Day 12: Boosting - AdaBoost
**Goal:** Understand boosting and sequential learning

**Watch (1h):**
- StatQuest: [AdaBoost](https://www.youtube.com/watch?v=LsK-xG1cLYA) (20m)
- StatQuest: [Gradient Boost Part 1](https://www.youtube.com/watch?v=3CC4N4z3GJc) (15m)
- StatQuest: [Gradient Boost Part 2](https://www.youtube.com/watch?v=jxuNLH5dXCs) (12m)

**Code from Scratch (3h):**
Implement AdaBoost:
- `DecisionStump` class (1-level tree)
- `AdaBoost` class with:
  - Initialize sample weights
  - Train weak learner with weights
  - Calculate weighted error
  - Update sample weights
  - Calculate stump weight
  - Weighted voting for prediction
- Test on `make_classification`

**Sklearn AdaBoost (1.5h):**
- Compare your implementation vs `AdaBoostClassifier`
- Try different base estimators
- Tune `n_estimators` and `learning_rate`
- Visualize: iteration vs training/validation error

**Deliverable:** `day12_adaboost.ipynb`

---

### Day 13: XGBoost & Modern Boosting
**Goal:** Master industry-standard boosting libraries

**Watch (45m):**
- StatQuest: [XGBoost Part 1](https://www.youtube.com/watch?v=OtD8wVaFm6E) (25m)
- StatQuest: [XGBoost Part 2](https://www.youtube.com/watch?v=8b1JEDvenQU) (25m)

**Install & Setup (15m):**
```bash
pip install xgboost lightgbm catboost
```

**XGBoost Basics (1.5h):**
Dataset: `load_breast_cancer`
- Create `DMatrix` for train/test
- Set parameters (max_depth, eta, objective)
- Train with `xgb.train()` and watchlist
- Plot training curves
- Feature importance (gain, cover, frequency)
- Early stopping

**Hyperparameter Tuning (2h):**
Use `RandomizedSearchCV` on `XGBClassifier`:
```python
{
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}
```

**Library Comparison (1h):**
Same dataset, same task:
- XGBoost
- LightGBM
- CatBoost
Compare: training time, accuracy, AUC

**Deliverable:** `day13_xgboost.ipynb`

---

### Day 14: PROJECT - Titanic Kaggle
**Goal:** Apply all tree models to real competition

**Competition:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

**EDA (1.5h):**
- Missing values heatmap
- Survival rate by: Sex, Pclass, Age groups, Embarked
- Correlation analysis
- Outlier detection
- Visualizations (seaborn)

**Feature Engineering (1.5h):**
- Handle missing: Age (median by Pclass), Embarked (mode), Fare
- Create features:
  - FamilySize = SibSp + Parch + 1
  - IsAlone (binary)
  - Title from Name (Mr, Mrs, Miss, Master, Other)
  - Age bins, Fare bins
- Encode: Sex, Embarked, Title
- Drop: Name, Ticket, Cabin
- Scaling: StandardScaler

**Modeling (2h):**
Try all models learned:
- Decision Tree
- Random Forest
- AdaBoost
- XGBoost
- LightGBM
- CatBoost

For each:
- 5-fold cross-validation
- Record: accuracy, precision, recall, AUC

Hyperparameter tune top 2 models
Create ensemble (VotingClassifier)

**Submission (30m):**
- Final predictions
- Create submission.csv
- Submit to Kaggle
- Document score

**Deliverable:** `day14_titanic.ipynb` + comprehensive README

---

## WEEK 3: OTHER ALGORITHMS & WORKFLOW (45 hours)

### Day 15: K-Nearest Neighbors
**Goal:** Distance-based learning

**Watch (20m):**
- StatQuest: [K-Nearest Neighbors](https://www.youtube.com/watch?v=HVXime0nQeI) (15m)

**Code from Scratch (2h):**
- Implement Euclidean distance
- Implement Manhattan distance
- Implement KNN class:
  - Store training data
  - Calculate distances to all points
  - Find k nearest neighbors
  - Majority vote (classification) or mean (regression)
- Test on iris dataset

**Sklearn Practice (2.5h):**
- `KNeighborsClassifier` on multiple datasets
- Try k values: `[1, 3, 5, 7, 11, 15, 21]`
- Plot: k vs accuracy
- Try different distance metrics
- Try different weights: `['uniform', 'distance']`
- Effect of feature scaling (with/without StandardScaler)

**Curse of Dimensionality (1h):**
- Experiment with high-dimensional data
- Show KNN performance degradation
- Discuss when to use/avoid KNN

**Deliverable:** `day15_knn.ipynb`

---

### Day 16: Naive Bayes
**Goal:** Probabilistic classification

**Watch (30m):**
- StatQuest: [Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA) (15m)

**Code from Scratch (2.5h):**
Implement Gaussian Naive Bayes:
- Calculate class priors
- Calculate mean and variance for each feature per class
- Calculate likelihood using Gaussian PDF
- Calculate posterior probabilities
- Predict using argmax
- Test on iris

**Sklearn Practice (2h):**
Try all Naive Bayes variants:
- `GaussianNB` on continuous data (iris, breast_cancer)
- `MultinomialNB` on count data (use CountVectorizer on text)
- `BernoulliNB` on binary data
- Compare performance

**Text Classification (1h):**
- Load 20newsgroups dataset (sklearn)
- Use `CountVectorizer` or `TfidfVectorizer`
- Apply `MultinomialNB`
- Evaluate with classification report

**Deliverable:** `day16_naive_bayes.ipynb`

---

### Day 17: Support Vector Machines
**Goal:** Maximum margin classifiers

**Watch (1h):**
- StatQuest: [Support Vector Machines Part 1](https://www.youtube.com/watch?v=efR1C6CvhmE) (20m)
- StatQuest: [Support Vector Machines Part 2](https://www.youtube.com/watch?v=Toet3EiSFcM) (20m)

**Theory Understanding (1h):**
- Read: https://scikit-learn.org/stable/modules/svm.html
- Understand: maximum margin, support vectors, kernel trick
- Draw margin diagrams
- **NO scratch implementation** (too complex mathematically)

**Sklearn Practice (3.5h):**
Dataset: `make_classification`, `make_circles`, `make_moons`

Linear SVM:
- `SVC(kernel='linear')`
- Visualize decision boundary and margins
- Plot support vectors

Kernel SVM:
- Try kernels: `['linear', 'poly', 'rbf', 'sigmoid']`
- Visualize decision boundaries for non-linear data
- Effect of C parameter: `[0.1, 1, 10, 100]`
- Effect of gamma (for rbf): `[0.001, 0.01, 0.1, 1]`

Real dataset (breast_cancer):
- GridSearchCV for best kernel and parameters
- Compare with other classifiers

SVR (Regression):
- Use `SVR` on diabetes dataset
- Tune epsilon and C

**Deliverable:** `day17_svm.ipynb`

---

### Day 18: Clustering - K-Means
**Goal:** Unsupervised learning introduction

**Watch (30m):**
- StatQuest: [K-Means Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA) (9m)

**Code from Scratch (2h):**
Implement K-Means:
- Random initialization of centroids
- Assignment step (assign points to nearest centroid)
- Update step (move centroids to cluster mean)
- Iterate until convergence
- Test on `make_blobs`
- Visualize iterations

**Sklearn Practice (2.5h):**
- `KMeans` on various datasets
- Elbow method (k vs inertia)
- Silhouette score analysis
- Compare different init methods: `['k-means++', 'random']`
- Effect of n_init

**Advanced Clustering (1h):**
- DBSCAN on non-spherical clusters
- Hierarchical clustering (dendrogram)
- Compare clustering methods

**Deliverable:** `day18_clustering.ipynb`

---

### Day 19: Feature Engineering Deep Dive
**Goal:** Advanced feature engineering techniques

**Watch (30m):**
- Various StatQuest videos on specific topics as needed

**Missing Data (1.5h):**
Dataset with lots of missing values:
- Simple imputation (mean, median, mode)
- KNN imputation
- Iterative imputation (MICE)
- Compare methods
- When to drop vs impute

**Encoding Techniques (1.5h):**
- One-hot encoding (`OneHotEncoder`)
- Label encoding (`LabelEncoder`)
- Ordinal encoding (`OrdinalEncoder`)
- Target encoding (manual implementation)
- Binary encoding
- When to use which

**Feature Creation (1.5h):**
- Polynomial features
- Interaction features
- Mathematical transformations (log, sqrt, power)
- Binning (equal width, equal frequency, custom)
- Date/time features (year, month, day, hour, day_of_week, is_weekend)

**Feature Selection (1h):**
- Variance threshold
- Univariate selection (SelectKBest)
- Recursive Feature Elimination (RFE)
- Feature importance from models
- L1-based selection

**Deliverable:** `day19_feature_engineering.ipynb`

---

### Day 20: Hyperparameter Tuning Mastery
**Goal:** Optimize model performance systematically

**Grid vs Random Search (1.5h):**
- `GridSearchCV` - exhaustive search
- `RandomizedSearchCV` - random sampling
- Compare time and performance
- When to use which

**Advanced Cross-Validation (1.5h):**
- K-Fold
- Stratified K-Fold (for imbalanced data)
- Time Series Split
- Group K-Fold
- Leave-One-Out (LOO)
- Nested cross-validation

**Pipeline Integration (2h):**
Build complete pipeline:
```python
Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest()),
    ('model', RandomForestClassifier())
])
```
- Tune all steps simultaneously
- Use `ColumnTransformer` for mixed data types

**Practical Tuning (1.5h):**
Real dataset with Pipeline:
- Define comprehensive param_grid
- RandomizedSearchCV with 100 iterations
- Analyze best parameters
- Learning curves
- Validation curves

**Deliverable:** `day20_hyperparameter_tuning.ipynb`

---

### Day 21: Imbalanced Data & Advanced Topics
**Goal:** Handle real-world data challenges

**Watch (30m):**
- StatQuest: Class Imbalance videos

**Imbalanced Data Techniques (2.5h):**
Create imbalanced dataset (90-10 split)

Techniques:
1. Class weights (`class_weight='balanced'`)
2. Undersampling majority class
3. Oversampling minority class
4. SMOTE (install: `pip install imbalanced-learn`)
5. ADASYN
6. Ensemble methods (EasyEnsemble, BalancedRandomForest)

Compare all methods:
- Accuracy (misleading!)
- Precision, Recall, F1
- ROC-AUC
- Confusion matrix

**Advanced Metrics (1.5h):**
- Precision-Recall curves
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Multi-class metrics (macro, micro, weighted)
- Custom scoring functions

**Model Interpretation (1.5h):**
Install: `pip install shap`
- Feature importance (tree-based models)
- Permutation importance
- SHAP values (basic introduction)
- Partial dependence plots

**Deliverable:** `day21_imbalanced_data.ipynb`

---

## WEEK 4: MAJOR PROJECTS (40 hours)

### Days 22-24: Major Project 1 - Classification
**Goal:** Build production-quality classification pipeline

**Choose Competition:**
- [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
- [Bank Marketing](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
- Any active Kaggle classification competition

**Day 22 - EDA & Planning (6h):**
- Deep exploratory analysis
- Missing value patterns
- Feature distributions
- Target analysis (check for imbalance)
- Correlation analysis
- Outlier detection
- Statistical tests
- Document insights in markdown
- Create feature engineering plan
- Create modeling strategy

**Day 23 - Feature Engineering & Baseline (6h):**
- Handle missing values (multiple strategies)
- Encode categorical variables
- Create new features (minimum 10)
- Feature scaling
- Feature selection experiments

Build baseline models:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Compare with cross-validation

**Day 24 - Advanced Modeling & Tuning (6h):**
- Hyperparameter tuning (top 3 models)
- Feature selection refinement
- Ensemble methods:
  - VotingClassifier
  - StackingClassifier
  - Blending
- Final model selection
- Test set predictions
- Kaggle submission

**Deliverables:**
- `project1_classification.ipynb` (well-documented)
- Comprehensive README with:
  - Problem description
  - Approach
  - Key insights from EDA
  - Feature engineering rationale
  - Model comparison
  - Final results
  - What worked/didn't work

---

### Days 25-27: Major Project 2 - Regression
**Goal:** Build end-to-end regression pipeline

**Choose Competition:**
- [House Prices - Advanced Regression](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Any active Kaggle regression competition

**Day 25 - EDA & Feature Engineering (6h):**
Similar rigor as Project 1:
- Comprehensive EDA
- Target variable analysis (skewness, outliers)
- Feature distributions
- Correlation analysis
- Missing value strategy
- Feature creation plan

Implement feature engineering:
- Handle missing values
- Log transform skewed features
- Create polynomial/interaction features
- Encode categorical variables
- Feature scaling

**Day 26 - Modeling & Validation (6h):**
Try multiple models:
- Linear models (Ridge, Lasso, ElasticNet)
- Tree models (RF, XGBoost, LightGBM, CatBoost)
- SVR (if appropriate)

For each model:
- Cross-validation (5-fold)
- Track: RMSE, MAE, R²
- Residual analysis
- Learning curves

**Day 27 - Tuning & Ensemble (6h):**
- Extensive hyperparameter tuning (top 3 models)
- Stacking multiple models
- Weighted averaging
- Final model selection based on CV scores
- Make predictions
- Submit to Kaggle
- Error analysis

**Deliverables:**
- `project2_regression.ipynb`
- Comprehensive README
- Comparison of all approaches
- Lessons learned document

---

### Day 28: Review, Portfolio & Preparation
**Goal:** Consolidate learning and prepare portfolio

**Morning - Review (3h):**
- Review all Anki cards created
- List topics that are still unclear
- Revisit 2-3 difficult concepts
- Create summary cheat sheet

**Afternoon - Portfolio (3h):**
- Clean all notebooks (remove experiments, add markdown)
- Write README for
