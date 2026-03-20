# Classification models compared
# Data source: Social_Network_Ads.csv
#
# This script recreates the main ideas from the original R workflow in Python:
# - quick visualization
# - logistic regression with threshold tuning
# - ROC / PR threshold exploration
# - random forest
# - XGBoost (basic, CV-tuned, and grid-searched)
# - KNN (with and without scaling)
# - simple neural net via sklearn
# - small Keras neural network
#
# The goal here is not to be ultra-fancy. It is to show how a few common
# classification approaches compare on a simple marketing / ad-response dataset.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier, cv as xgb_cv, DMatrix
import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# 1. Load the data
# -----------------------------
# Put Social_Network_Ads.csv in the same folder as this script,
# or replace the path below with the correct location.
df = pd.read_csv("Social_Network_Ads.csv")

print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nShape:")
print(df.shape)


# -----------------------------
# 2. Keep only the columns used in the R code
# -----------------------------
# The original analysis focused on Age and EstimatedSalary as predictors
# and Purchased as the target.
df = df[["Age", "EstimatedSalary", "Purchased"]].copy()

print("\nClass balance:")
print(df["Purchased"].value_counts(normalize=True))


# -----------------------------
# 3. Quick scatter plot
# -----------------------------
# This gives a simple visual sense of how purchase behavior relates
# to age and salary.
plt.figure(figsize=(8, 6))

not_purchased = df[df["Purchased"] == 0]
purchased = df[df["Purchased"] == 1]

plt.scatter(
    not_purchased["Age"],
    not_purchased["EstimatedSalary"],
    alpha=0.7,
    s=50,
    label="Not Purchased"
)
plt.scatter(
    purchased["Age"],
    purchased["EstimatedSalary"],
    alpha=0.7,
    s=50,
    label="Purchased"
)

plt.title("Age vs Salary by Purchase Status")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.tight_layout()
plt.show()

# The rough visual story is similar to what you noted in R:
# people who purchase tend to skew older and higher-income.


# -----------------------------
# 4. Train/test split
# -----------------------------
# Same basic idea as createDataPartition(..., p = 0.7) in caret.
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=123,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))


# -----------------------------
# Helper functions
# -----------------------------
def print_model_results(name, y_true, y_pred):
    """Print a compact set of evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    print(f"\n{name}")
    print("-" * len(name))
    print("Confusion matrix:")
    print(cm)
    print(f"Accuracy:    {acc:.3f}")
    print(f"Precision:   {prec:.3f}")
    print(f"Recall:      {rec:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 score:    {f1:.3f}")


def best_balanced_threshold_from_roc(y_true, probs):
    """
    Find the ROC threshold where sensitivity and specificity are as balanced as possible.
    This is the Python equivalent of chasing that 'best' threshold from the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    specificity = 1 - fpr
    diff = np.abs(tpr - specificity)
    best_idx = np.argmin(diff)

    return {
        "threshold": thresholds[best_idx],
        "sensitivity": tpr[best_idx],
        "specificity": specificity[best_idx]
    }


def best_balanced_threshold_from_pr(y_true, probs):
    """
    Find the PR threshold where precision and recall are closest to each other.
    This mirrors what you were doing in R with the PR curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    # precision_recall_curve returns one extra precision/recall point compared to thresholds,
    # so we align them by dropping the last precision/recall entry.
    precision = precision[:-1]
    recall = recall[:-1]

    diff = np.abs(precision - recall)
    best_idx = np.argmin(diff)

    return {
        "threshold": thresholds[best_idx],
        "precision": precision[best_idx],
        "recall": recall[best_idx]
    }


# -----------------------------
# 5. Logistic regression
# -----------------------------
# Logistic regression can work well here because the problem is small and the
# relationship is pretty interpretable.
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

log_probs = log_model.predict_proba(X_test)[:, 1]

# Try the same three thresholds you were comparing in R.
for threshold in [0.5, 0.311, 0.405]:
    log_preds = (log_probs > threshold).astype(int)
    print_model_results(
        f"Logistic Regression (threshold = {threshold})",
        y_test,
        log_preds
    )

# AUC is also useful here as an overall summary of ranking quality.
fpr, tpr, roc_thresholds = roc_curve(y_test, log_probs)
roc_auc = auc(fpr, tpr)

print(f"\nLogistic Regression ROC AUC: {roc_auc:.3f}")

# Plot ROC curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.show()

# Find the threshold that gives the best balance of sensitivity and specificity.
roc_best = best_balanced_threshold_from_roc(y_test, log_probs)

print("\nBest balanced threshold from ROC:")
print(f"Threshold:   {roc_best['threshold']:.6f}")
print(f"Sensitivity: {roc_best['sensitivity']:.3f}")
print(f"Specificity: {roc_best['specificity']:.3f}")

# Now do the PR version too, since the positive class is a bit smaller.
pr_best = best_balanced_threshold_from_pr(y_test, log_probs)

print("\nBest balanced threshold from PR curve:")
print(f"Threshold: {pr_best['threshold']:.6f}")
print(f"Precision: {pr_best['precision']:.3f}")
print(f"Recall:    {pr_best['recall']:.3f}")

# Plot PR curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, log_probs)

plt.figure(figsize=(7, 5))
plt.plot(recall_vals, precision_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Logistic Regression")
plt.tight_layout()
plt.show()


# -----------------------------
# 6. Random forest
# -----------------------------
# Tree ensembles are usually a strong choice for small tabular problems like this.
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_features=2,   # same spirit as mtry = 2 in R
    random_state=1
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

print_model_results("Random Forest", y_test, rf_preds)


# -----------------------------
# 7. XGBoost - basic model
# -----------------------------
# XGBoost often does very well on structured/tabular datasets.
xgb_model = XGBClassifier(
    n_estimators=100,
    objective="binary:logistic",
    max_depth=3,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=123
)

xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# Use the same threshold you were exploring in R.
xgb_preds = (xgb_probs > 0.311).astype(int)

print_model_results("XGBoost (basic, threshold = 0.311)", y_test, xgb_preds)


# -----------------------------
# 8. XGBoost with CV to choose number of rounds
# -----------------------------
# In the R version, you used xgb.cv to find a better number of boosting rounds.
# Here we do the same with the native xgboost cv function.
dtrain = DMatrix(X_train, label=y_train)

params = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "eta": 0.1,
    "eval_metric": "auc",
    "seed": 123
}

cv_results = xgb_cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=200,
    nfold=5,
    stratified=True,
    early_stopping_rounds=10,
    verbose_eval=False
)

best_nrounds = len(cv_results)
print(f"\nBest number of boosting rounds from CV: {best_nrounds}")

xgb_model_cv = XGBClassifier(
    n_estimators=best_nrounds,
    objective="binary:logistic",
    max_depth=3,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=123
)

xgb_model_cv.fit(X_train, y_train)
xgb_probs_cv = xgb_model_cv.predict_proba(X_test)[:, 1]
xgb_preds_cv = (xgb_probs_cv > 0.311).astype(int)

print_model_results("XGBoost (CV-tuned rounds, threshold = 0.311)", y_test, xgb_preds_cv)


# -----------------------------
# 9. XGBoost grid search
# -----------------------------
# This automates the hyperparameter tweaking instead of doing it manually.
xgb_grid_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="error",
    random_state=123
)

param_grid = {
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "min_child_weight": [1, 5],
    "n_estimators": [50, 100, 150, 200]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

grid_search = GridSearchCV(
    estimator=xgb_grid_model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

print("\nBest XGBoost grid-search parameters:")
print(grid_search.best_params_)
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")

best_xgb_grid = grid_search.best_estimator_
xgb_probs_grid = best_xgb_grid.predict_proba(X_test)[:, 1]

# Check test accuracy across a range of thresholds, like you did in R.
thresholds = np.arange(0.30, 0.71, 0.01)
accs = []

for t in thresholds:
    preds_t = (xgb_probs_grid > t).astype(int)
    accs.append(accuracy_score(y_test, preds_t))

plt.figure(figsize=(7, 5))
plt.plot(thresholds, accs)
plt.title("XGBoost Grid Search: Accuracy vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

# Use a threshold close to the one you were favoring.
xgb_preds_grid = (xgb_probs_grid > 0.3111).astype(int)
print_model_results("XGBoost (grid search, threshold = 0.3111)", y_test, xgb_preds_grid)


# -----------------------------
# 10. KNN without scaling
# -----------------------------
# KNN is very sensitive to scale, so this is mostly here to show why preprocessing matters.
knn_unscaled = KNeighborsClassifier()

knn_param_grid = {"n_neighbors": list(range(1, 21))}
knn_grid_unscaled = GridSearchCV(
    knn_unscaled,
    knn_param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

knn_grid_unscaled.fit(X_train, y_train)
knn_unscaled_best = knn_grid_unscaled.best_estimator_
knn_preds_unscaled = knn_unscaled_best.predict(X_test)

print(f"\nBest KNN k without scaling: {knn_grid_unscaled.best_params_['n_neighbors']}")
print_model_results("KNN without scaling", y_test, knn_preds_unscaled)


# -----------------------------
# 11. KNN with scaling
# -----------------------------
# This is the more fair version of KNN because salary is on a much larger scale than age.
knn_scaled_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

knn_scaled_param_grid = {
    "knn__n_neighbors": list(range(1, 21))
}

knn_grid_scaled = GridSearchCV(
    knn_scaled_pipeline,
    knn_scaled_param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

knn_grid_scaled.fit(X_train, y_train)
knn_scaled_best = knn_grid_scaled.best_estimator_
knn_preds_scaled = knn_scaled_best.predict(X_test)

print(f"\nBest KNN k with scaling: {knn_grid_scaled.best_params_['knn__n_neighbors']}")
print_model_results("KNN with scaling", y_test, knn_preds_scaled)


# -----------------------------
# 12. Simple neural net with sklearn
# -----------------------------
# This is a lightweight substitute for the small neuralnet model you had in R.
# We scale first because neural nets usually behave much better when predictors
# are on similar ranges.
nn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(5,),
        activation="relu",
        max_iter=2000,
        random_state=123
    ))
])

nn_pipeline.fit(X_train, y_train)
nn_preds = nn_pipeline.predict(X_test)

print_model_results("Simple Neural Net (sklearn MLP)", y_test, nn_preds)


# -----------------------------
# 13. Keras neural network
# -----------------------------
# This mirrors your later Keras experiment more closely.
# Again, scale first because neural nets care about that a lot.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Make the split for Keras validation logic a little cleaner.
# We keep it simple here and use validation_split during fit.
keras_model = keras.Sequential([
    layers.Dense(4, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(1, activation="sigmoid")
])

keras_model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = keras_model.fit(
    X_train_scaled,
    y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

keras_probs = keras_model.predict(X_test_scaled, verbose=0).flatten()
keras_preds = (keras_probs > 0.5).astype(int)

print_model_results("Keras Neural Net", y_test, keras_preds)


# -----------------------------
# 14. Final summary table
# -----------------------------
# This makes it easier to compare everything side by side.
def collect_metrics(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }


summary_rows = [
    collect_metrics("Logistic (0.5)", y_test, (log_probs > 0.5).astype(int)),
    collect_metrics("Logistic (0.311)", y_test, (log_probs > 0.311).astype(int)),
    collect_metrics("Logistic (0.405)", y_test, (log_probs > 0.405).astype(int)),
    collect_metrics("Random Forest", y_test, rf_preds),
    collect_metrics("XGBoost basic", y_test, xgb_preds),
    collect_metrics("XGBoost CV rounds", y_test, xgb_preds_cv),
    collect_metrics("XGBoost grid", y_test, xgb_preds_grid),
    collect_metrics("KNN unscaled", y_test, knn_preds_unscaled),
    collect_metrics("KNN scaled", y_test, knn_preds_scaled),
    collect_metrics("Neural Net (sklearn)", y_test, nn_preds),
    collect_metrics("Keras Neural Net", y_test, keras_preds),
]

summary_df = pd.DataFrame(summary_rows).sort_values(by="Accuracy", ascending=False)

print("\nModel comparison summary:")
print(summary_df.round(3).to_string(index=False))