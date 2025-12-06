# Blinking-ML
Just getting started again 
---

**Total: 130-150 hours | 5-6 hours/day | 5 days/week (Mon-Fri) + catch-up weekends**

### **DAILY STRUCTURE:**
- **Math warmup (20 min):** 3Blue1Brown or quick practice problems
- **Main work (4-5 hours):** Study + code + practice
- **Review (30 min):** Update GitHub, document learnings
- **Saturdays:** Catch-up or rest
- **Sundays:** Project work or complete rest

---

### **WEEK 1: LINEAR MODELS (25-30 hours)**

**Day 1: Linear Regression Deep Dive**  
*Goal: Build intuition and implement from scratch*

**Math warmup (20m):** 3Blue1Brown - Chapter 1 (Vectors)

**Watch (30m):**
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo) (9m) - Yes, old but clearest explanation
- [ritvikmath: Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8) (14m) - 2021, better visuals

**Code from Scratch (2.5h):**
```python
# Implement these functions:
1. generate_data(n_samples, noise) 
2. cost_function(X, y, theta)
3. gradient_descent(X, y, learning_rate, iterations)
4. plot_regression_line()
5. plot_cost_history()
```
- Generate: `y = 4 + 3x + noise`
- Implement gradient descent
- Visualize convergence
- **Reference ONLY if stuck >30min:** [GitHub examples](https://github.com/topics/linear-regression-python)

**Sklearn Practice (1.5h):**
- [Getting Started Tutorial](https://scikit-learn.org/stable/getting_started.html) - Execute every example
- Compare your implementation vs `sklearn.linear_model.LinearRegression`
- Calculate MSE, R², visualize predictions vs actual

**Deliverable:** `day1_linear_regression.ipynb` + push to GitHub

---

**Day 2: Multiple Regression & Vectorization**  
*Goal: Master matrix operations*

**Math warmup (20m):** 3Blue1Brown - Chapter 2 (Linear combinations)

**Watch (20m):**
- [StatQuest: Multiple Regression](https://www.youtube.com/watch?v=zITIFTsivN8) (8m)
- Skip extra videos, jump to coding

**Code from Scratch (2h):**
```python
# NO LOOPS in training loop
1. Vectorized gradient descent with NumPy
2. Normal equation: θ = (X^T X)^-1 X^T y
3. Compare speed (use time.time())
4. Test on sklearn.datasets.make_regression (5 features)
```

**Real Dataset (2h):**
- Load: `fetch_california_housing()` 
- Train/test split (80-20)
- Apply both methods
- Error analysis: residuals plot, feature correlations
- Which features matter most?

**Deliverable:** `day2_vectorization.ipynb`

---

**Day 3: Regularization (Ridge/Lasso)**  
*Goal: Prevent overfitting*

**Math warmup (20m):** 3Blue1Brown - Chapter 3

**Watch (30m):**
- [StatQuest: Ridge Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) (20m)
- [StatQuest: Lasso Regression](https://www.youtube.com/watch?v=NGf0voTMlcs) (8m)

**Theory Practice (1h):**
- Create dataset with **highly correlated features**
- Manually create correlation: `X2 = X1 + small_noise`
- Show OLS fails (high coefficients)

**Sklearn Deep Dive (2.5h):**
```python
# Try alphas: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# For each alpha:
# - Train model
# - 5-fold cross-validation
# - Plot: alpha vs CV score
# - Plot: alpha vs coefficient magnitudes
```

**Advanced (1h):**
- `RidgeCV`, `LassoCV` (built-in cross-validation)
- When to use Ridge vs Lasso vs ElasticNet?
- Document findings in markdown

**Deliverable:** `day3_regularization.ipynb`

---

**Day 4: Logistic Regression**  
*Goal: Binary classification*

**Math warmup (20m):** 3Blue1Brown - Chapter 4

**Watch (30m):**
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) (9m)
- [3Blue1Brown: Neural Networks Ch1](https://www.youtube.com/watch?v=aircAruvnKk) (19m) - Watch 0:00-8:00 for sigmoid intuition

**Code from Scratch (2h):**
```python
# Implement:
1. sigmoid(z)
2. binary_cross_entropy_loss(y_true, y_pred)
3. logistic_gradient_descent(X, y, learning_rate, iterations)
4. predict_proba(X, theta)
5. predict_class(X, theta, threshold=0.5)

# Test on: make_classification(n_features=2) for 2D visualization
# Plot: decision boundary
```

**Sklearn Practice (2h):**
```python
# Dataset: load_breast_cancer()
# Try:
solvers = ['lbfgs', 'liblinear', 'saga']
penalties = ['l1', 'l2', 'elasticnet', None]
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

# For each combination:
# - Cross-validation
# - Compare with your implementation
# - Visualize decision boundaries (2 features only)
```

**Deliverable:** `day4_logistic_regression.ipynb`

---

**Day 5: Classification Metrics Mastery**  
*Goal: Beyond accuracy*

**Math warmup (20m):** 3Blue1Brown - Chapter 5

**Watch (45m):**
- [StatQuest: Sensitivity/Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8) (12m)
- [StatQuest: ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM) (17m)
- [StatQuest: Precision and Recall](https://www.youtube.com/watch?v=Kdsp6soqA7o) (6m)

**Code from Scratch (1.5h):**
```python
# Implement these (then verify with sklearn):
1. confusion_matrix(y_true, y_pred)
2. accuracy, precision, recall, f1_score
3. ROC curve: 
   - Try all thresholds
   - Calculate TPR, FPR for each
   - Plot curve
4. AUC calculation (trapezoidal rule)
```

**Sklearn Metrics (2h):**
```python
# Dataset: breast_cancer
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_curve, roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# Create:
# 1. Confusion matrix heatmap (seaborn)
# 2. ROC curve with AUC
# 3. Precision-Recall curve with AP
# 4. Classification report
# 5. Try different thresholds: [0.3, 0.5, 0.7]
#    - When would you use each?
```

**Read (1h):**
- [Classification Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) - Read carefully, this is critical

**Deliverable:** `day5_metrics.ipynb`

---

### **WEEKEND 1 (Saturday):**

**Option A - Catch up** if any day took longer  
**Option B - Mini Project:**
- Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Highly imbalanced (use what you learned)
- Apply logistic regression with proper metrics
- 2-3 hours max, keep it simple

---

### **WEEK 2: TREE MODELS (25-30 hours)**

**Day 6: Decision Trees (Theory + Scratch)**

**Math warmup (20m):** 3Blue1Brown Calculus - Chapter 1

**Watch (30m):**
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk) (17m)
- [StatQuest: Decision Trees Part 2](https://www.youtube.com/watch?v=wpNl-JwwplA) (10m)

**Code from Scratch (3h):**
```python
# Build simple CART tree (Classification And Regression Tree)
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        ...

def gini_index(y): ...
def split_data(X, y, feature, threshold): ...
def best_split(X, y): # Try all features & thresholds
def build_tree(X, y, depth=0, max_depth=5): # Recursive
def predict_sample(x, node): ...
def predict(X, tree): ...
```
- Test on `load_iris()`
- Visualize tree structure (print statements)
- **This is important - you MUST do this from scratch**

**Sklearn Trees (1.5h):**
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Try max_depths: [2, 3, 5, 10, None]
# Visualize each with plot_tree
# Plot: depth vs accuracy (train and validation)
# Understand overfitting visually
```

**Deliverable:** `day6_decision_trees.ipynb`

---

**Day 7: Random Forest (Concept + Sklearn)**

**Math warmup (20m):** 3Blue1Brown Calculus - Chapter 2

**Watch (40m):**
- [StatQuest: Random Forests Part 1](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) (10m)
- [StatQuest: Random Forests Part 2](https://www.youtube.com/watch?v=nyxTdL_4Q-Q) (14m)
- [StatQuest: Feature Selection](https://www.youtube.com/watch?v=YaKMeAlHgqQ) (14m)

**Conceptual Implementation (1.5h):**
```python
# Build SIMPLE random forest (don't go crazy):
1. bootstrap_sample(X, y) # Sampling with replacement
2. Train 10 decision trees with random feature subsets
3. Majority voting for predictions
4. Compare: 1 tree vs 5 trees vs 10 trees

# This is just for understanding - sklearn does this better
```

**Sklearn Deep Dive (2.5h):**
```python
# Dataset: load_digits() or fetch_covtype() (harder)
from sklearn.ensemble import RandomForestClassifier

# Experiment:
n_estimators = [10, 50, 100, 200, 500]
max_depth = [5, 10, 20, None]
min_samples_split = [2, 5, 10]
max_features = ['sqrt', 'log2', None]

# Use RandomizedSearchCV (50 iterations)
# Plot: n_estimators vs accuracy
# Feature importance analysis
# OOB score vs validation score
```

**Advanced (1h):**
- `ExtraTreesClassifier` - what's different?
- Compare RF vs single tree on same data
- When does RF fail?

**Deliverable:** `day7_random_forest.ipynb`

---

**Day 8: Gradient Boosting Theory**

**Math warmup (20m):** 3Blue1Brown Calculus - Chapter 3

**Watch (1.5h):**
- [StatQuest: Gradient Boost Part 1](https://www.youtube.com/watch?v=3CC4N4z3GJc) (15m)
- [StatQuest: Gradient Boost Part 2](https://www.youtube.com/watch?v=2xudPOBz-vs) (12m)
- [StatQuest: Gradient Boost Part 3](https://www.youtube.com/watch?v=jxuNLH5dXCs) (12m)
- [StatQuest: Gradient Boost Part 4](https://www.youtube.com/watch?v=StWY5QWMXCw) (14m)

**Conceptual Understanding (2h):**
- **DO NOT implement from scratch** (too complex, not worth it)
- Instead, manually work through paper example:
  - Start with mean prediction
  - Calculate residuals
  - Fit tree to residuals
  - Update predictions
  - Repeat
- Draw diagrams, understand the algorithm deeply

**Sklearn GradientBoosting (2h):**
```python
from sklearn.ensemble import GradientBoostingClassifier

# Dataset: breast_cancer
# Key parameters:
n_estimators = [50, 100, 200]
learning_rate = [0.01, 0.05, 0.1, 0.3]
max_depth = [3, 5, 7]
subsample = [0.8, 1.0]

# For each:
# - Train with early stopping
# - Plot: iteration vs train/val loss
# - Understand learning_rate vs n_estimators tradeoff
```

**Deliverable:** `day8_gradient_boosting.ipynb`

---

**Day 9: XGBoost, LightGBM, CatBoost**

**Math warmup (20m):** 3Blue1Brown Calculus - Chapter 4

**Install (10m):**
```bash
pip install xgboost lightgbm catboost
```

**Watch (1h):**
- [StatQuest: XGBoost Part 1](https://www.youtube.com/watch?v=OtD8wVaFm6E) (25m)
- [StatQuest: XGBoost Part 2](https://www.youtube.com/watch?v=8b1JEDvenQU) (25m)

**XGBoost Practice (2h):**
```python
import xgboost as xgb

# Dataset: load_breast_cancer()
# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters to understand:
params = {
    'max_depth': 6,
    'eta': 0.3,  # learning_rate
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

# Train with watchlist (see training progress)
# Plot feature importance (gain, cover, frequency)
# Early stopping
# Learning curves
```

**Library Comparison (2h):**
```python
# Same dataset, same task
# Compare: XGBoost vs LightGBM vs CatBoost

# Measure:
# - Training time
# - Prediction time
# - AUC score
# - Memory usage (rough estimate)
# - Feature importance consistency

# When to use each?
```

**Deliverable:** `day9_modern_boosting.ipynb`

---

**Day 10: Hyperparameter Tuning for Trees**

**Math warmup (20m):** 3Blue1Brown Calculus - Chapter 5

**Watch (30m):**
- [Hyperparameter Tuning Overview](https://www.youtube.com/watch?v=5nYqK-HaoKY) (newer resource, search "hyperparameter tuning machine learning 2023")

**Sklearn Tuning Tools (2h):**
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Dataset: A challenging one like load_digits()

# Grid Search (exhaustive, slower):
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro')

# Randomized Search (faster):
param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10, 20]
}
random_search = RandomizedSearchCV(rf, param_dist, n_iter=50, cv=5)

# Compare time and results
```

**Modern Tuning - Optuna (2h):**
```python
# Install: pip install optuna
import optuna

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    model = xgb.XGBClassifier(**param)
    score = cross_val_score(model, X, y, cv=5, scoring='f1_macro').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Visualize optimization history
# Much faster than Grid/Random Search
```

**Deliverable:** `day10_hyperparameter_tuning.ipynb`

---

### **WEEKEND 2 (Saturday/Sunday):**

**MINI-PROJECT: Kaggle Competition Start**

Choose ONE:
- [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
- [Playground Series - latest competition](https://www.kaggle.com/competitions?hostSegmentIdFilter=8)

**Sunday (4-6 hours):**
- Full EDA
- Baseline models: Logistic, RF, XGBoost
- At least one submission
- This is practice for bigger projects ahead

---

### **WEEK 3: OTHER ALGORITHMS + WORKFLOW (25-30 hours)**

**Day 11: K-Nearest Neighbors**

**Math warmup (20m):** Review linear algebra concepts

**Watch (20m):**
- [StatQuest: KNN](https://www.youtube.com/watch?v=HVXime0nQeI) (15m)

**Code from Scratch (2h):**
```python
# Implement:
def euclidean_distance(x1, x2): ...
def manhattan_distance(x1, x2): ...

class KNN:
    def __init__(self, k=3, distance='euclidean'):
        ...
    def fit(X, y): # Just store training data
    def predict(X): # For each point, find k nearest neighbors
    
# Test on iris
# Visualize decision boundaries (2D)
```

**Sklearn Practice (2.5h):**
```python
from sklearn.neighbors import KNeighborsClassifier

# Test on multiple datasets
# Try k = [1, 3, 5, 7, 11, 15, 21, 31]
# Plot: k vs accuracy (train and val)
# Distance metrics: ['euclidean', 'manhattan', 'minkowski']
# Weights: ['uniform', 'distance']

# Critical: Effect of scaling
# - Train KNN without StandardScaler
# - Train KNN with StandardScaler
# - Compare (HUGE difference expected)
```

**Curse of Dimensionality (1h):**
```python
# Create datasets with increasing dimensions
# Show KNN performance degradation
# When to use/avoid KNN?
```

**Deliverable:** `day11_knn.ipynb`

---

**Day 12: Support Vector Machines**

**Math warmup (20m):** Review dot products, projections

**Watch (1h):**
- [StatQuest: SVM Part 1](https://www.youtube.com/watch?v=efR1C6CvhmE) (20m)
- [StatQuest: SVM Part 2](https://www.youtube.com/watch?v=Toet3EiSFcM) (20m)
- [StatQuest: Kernel Trick](https://www.youtube.com/watch?v=Qc5IyLW_hns) (8m)

**Theory (1h):**
- Read: https://scikit-learn.org/stable/modules/svm.html
- Understand: maximum margin, support vectors, kernel trick
- Draw diagrams, understand intuitively
- **Skip scratch implementation** (math too complex, not worth time)

**Sklearn Practice (3h):**
```python
from sklearn.svm import SVC, SVR

# Start with synthetic data:
# - make_classification (linearly separable)
# - make_circles (non-linear)
# - make_moons (non-linear)

# Linear SVM:
svm = SVC(kernel='linear', C=1.0)
# Plot decision boundary and margin
# Plot support vectors (highlight them)

# Kernel SVM:
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# Visualize decision boundaries for each
# Effect of C: [0.1, 1, 10, 100]
# Effect of gamma (rbf): [0.001, 0.01, 0.1, 1]

# Real dataset (breast_cancer):
# GridSearchCV for best kernel and params
# Compare with other classifiers

# SVR on diabetes dataset
```

**Deliverable:** `day12_svm.ipynb`

---

**Day 13: Naive Bayes**

**Math warmup (20m):** Review probability basics

**Watch (30m):**
- [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA) (15m)

**Code from Scratch (2h):**
```python
# Implement Gaussian Naive Bayes:
class GaussianNB:
    def fit(X, y):
        # Calculate class priors
        # Calculate mean and std for each feature per class
    
    def predict(X):
        # For each class, calculate likelihood using Gaussian PDF
        # Calculate posterior = prior * likelihood
        # Return class with highest posterior

# Test on iris
```

**Sklearn Practice (2.5h):**
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# GaussianNB: continuous data (iris, breast_cancer)
# MultinomialNB: count data (text)
# BernoulliNB: binary data

# Text Classification:
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load 2-3 categories
# Vectorize text
# Apply MultinomialNB
# Classification report
# Compare CountVectorizer vs TfidfVectorizer
```

**When to use NB? (1h):**
- Advantages: fast, works with small data, handles high dimensions
- Disadvantages: independence assumption
- Compare NB vs Logistic Regression on same data

**Deliverable:** `day13_naive_bayes.ipynb`

---

**Day 14: Clustering (K-Means + others)**

**Math warmup (20m):** Review distance metrics

**Watch (30m):**
- [StatQuest: K-Means](https://www.youtube.com/watch?v=4b5d3muPQmA) (9m)
- [StatQuest: Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo) (12m)

**Code from Scratch (1.5h):**
```python
# Implement K-Means:
class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        ...
    
    def fit(X):
        # Random initialization
        # Assignment step
        # Update step
        # Iterate until convergence
        # Track inertia

# Test on make_blobs
# Visualize iterations (create animation if possible)
```

**Sklearn Clustering (2.5h):**
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# K-Means:
# - Elbow method (k vs inertia)
# - Silhouette score (k vs silhouette)
# - Effect of init: ['k-means++', 'random']
# - Effect of n_init

# DBSCAN (density-based):
# - Test on non-spherical clusters
# - Tune eps and min_samples
# - Handles outliers, no need to specify k

# Hierarchical:
# - Dendrogram visualization
# - Different linkages: ['ward', 'complete', 'average']

# Compare all three methods on same dataset
```

**Deliverable:** `day14_clustering.ipynb`

---

**Day 15: Dimensionality Reduction (PCA + t-SNE)**

**Math warmup (20m):** Review eigenvectors (high-level)

**Watch (40m):**
- [StatQuest: PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ) (21m)
- [StatQuest: t-SNE](https://www.youtube.com/watch?v=NEaUSP4YerM) (12m)

**PCA Deep Dive (2.5h):**
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load MNIST-like data (digits)
# Original: 64 dimensions

# PCA:
# - Fit PCA with all components
# - Plot explained variance ratio
# - Cumulative explained variance (how many components for 95%?)
# - Reduce to 2D, visualize with scatter plot (color by digit)
# - Reconstruct images from reduced dimensions
# - Visual comparison: original vs reconstructed

# Try on other datasets (breast_cancer, wine)
```

**t-SNE Practice (1.5h):**
```python
from sklearn.manifold import TSNE

# t-SNE for visualization only (not for training)
# Reduce digits to 2D
# Compare t-SNE vs PCA visualization
# Tune perplexity: [5, 30, 50, 100]
# Warning: t-SNE is SLOW, use PCA first to reduce to 50D, then t-SNE
```

**When to use what? (1h):**
- PCA: linear, preserves global structure, fast
- t-SNE: non-linear, preserves local structure, slow, visualization only
- Document use cases

**Deliverable:** `day15_dimensionality_reduction.ipynb`

---

### **WEEKEND 3 (Saturday/Sunday):**

**Study/Rest Day + Review**

**Saturday (3-4 hours):**
- Review all notebooks from Week 3
- Create comparison charts (when to use each algorithm)
- Update GitHub README with progress
- Identify weak topics, revisit

**Sunday:** Complete rest OR catch up if behind

---

### **WEEK 4: MAJOR PROJECT (25-30 hours)**

**Days 16-20: One Killer Kaggle Project**

Choose ONE competition (pick based on interest):
- [House Prices - Advanced Regression](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
- Any active [Playground Series competition](https://www.kaggle.com/competitions?hostSegmentIdFilter=8)

**Day 16: Deep EDA (5-6 hours)**

```python
# Comprehensive analysis:
1. Load data, understand problem
2. Target variable analysis
   - Distribution (histogram, box plot)
   - Check for skewness
   - Outliers
3. Feature analysis
   - Data types (numerical vs categorical)
   - Missing values (heatmap, percentage)
   - Distributions (histograms for all features)
   - Unique values for categorical
4. Relationships
   - Correlation matrix (heatmap)
   - Target vs each feature (scatter, box plots)
   - Categorical feature analysis (count plots)
5. Statistical tests
   - Check feature-target relationships
6. Insights
   - Document all findings in markdown
   - What features look promising?
   - What needs engineering?
   - What's the strategy?
```

**Day 17: Feature Engineering (5-6 hours)**

```python
# Missing values
1. Analyze missing patterns
2. Decide strategy for each:
   - Mean/median (numerical)
   - Mode (categorical)
   - Forward/backward fill (time series)
   - Drop (if too many missing)
   - Indicator variable (missing or not)

# Feature creation (aim for 15-20 new features)
1. Polynomial features (degree 2)
2. Interaction terms (important feature pairs)
3. Mathematical transformations
   - Log (skewed features)
   - Square root, square
   - Binning (age groups, etc.)
4. Categorical encoding
   - One-hot (low cardinality)
   - Label encoding (ordinal)
   - Target encoding (high cardinality)
5. Domain-specific features
   - For House Prices: TotalSF, Age, Quality scores
   - For Spaceship: GroupSize, family features
6. Aggregations (if applicable)

# Feature selection
1. Drop high correlation (>0.95)
2. Variance threshold
3. Feature importance from basic model
4. Keep top features

# Scaling
1. StandardScaler for tree-free models
2. No scaling for tree-based models
```

**Day 18: Baseline Models (5-6 hours)**

```python
# Try 6-8 models with default parameters:
1. Logistic Regression / Linear Regression
2. Ridge / Lasso
3. Decision Tree
4. Random Forest
5. XGBoost
6. LightGBM
7. CatBoost (if categorical features)
8. SVM (if small dataset)

# For each model:
# - 5-fold cross-validation
# - Record: mean CV score, std, training time
# - Create comparison table

# Select top 3 for hyperparameter tuning
# Analyze: which features are important? (tree-based models)
# Error analysis: residuals, predictions vs actual
```

**Day 19: Hyperparameter Tuning + Ensemble (6-7 hours)**

```python
# Tune top 3 models
# Use Optuna or RandomizedSearchCV

# Example for XGBoost:
param_dist = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# Run 100+ iterations
# For each tuned model, retrain with best params

# Ensemble methods:
1. VotingRegressor/Classifier (average predictions)
2. StackingRegressor/Classifier
   - Use tuned models as base estimators
   - Use linear model as final estimator
3. Weighted averaging (manual)
   - Weight based on CV scores

# Compare: best single model vs ensemble
```

**Day 20: Final Submission + Documentation (4-5 hours)**

```python
# Final model selection
1. Based on CV scores
2. Train on full training set
3. Predict on test set
4. Create submission.csv
5. Submit to Kaggle

# If time permits:
# - Try different feature engineering
# - Create 2nd submission with different approach
# - Ensemble your own submissions

# Documentation (README.md):
1. Problem description
2. Approach
   - EDA insights
   - Feature engineering rationale
   - Models tried
   - Tuning strategy
3. Results
   - CV scores vs leaderboard score
   - What worked
   - What didn't work
   - Learnings
4. Next steps (what would you try with more time?)

# Code cleanup:
# - Remove experimental cells
# - Add markdown explanations
# - Ensure reproducibility
```

**Target: Top 40-50% on leaderboard** (realistic for first major project)

---

### **Final Days (21-22): Review + Portfolio**

**Day 21: Code Review + GitHub (3-4 hours)**

```python
# For each notebook:
1. Add comprehensive markdown
2. Remove experimental/failed code
3. Add comments
4. Ensure reproducibility
5. Create requirements.txt

# GitHub organization:
week1_linear_models/
week2_tree_models/
week3_other_algorithms/
week4_major_project/
README.md (main, with links to all weeks)
```

**Day 22: Learning Reflection (2-3 hours)**

```markdown
# Create document: month1_learnings.md

1. What I learned
   - Technical skills
   - Algorithms understood
   - Tools mastered

2. What was difficult
   - Which concepts took longest
   - What needs more practice

3. What surprised me
   - Unexpected insights
   - Performance discoveries

4. Month 2 goals
   - What to focus on
   - What to skip
   - How to improve

5. Interview prep
   - Can I explain each algorithm?
   - Can I code basics from scratch?
   - Do I know when to use what?
```
