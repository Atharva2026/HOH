
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix
from data_cleaner import get_cleaned_processed_data

# Set random seed
RANDOM_STATE = 42

# --------------------------------------------
# 1. Load & Process Data
# --------------------------------------------
print("Loading, Cleaning, and Engineering Features...")
X, y = get_cleaned_processed_data("data.csv")

if X is None:
    exit()

print(f"Data Shape after Feature Engineering: {X.shape}")

# Convert categoricals to 'category' dtype for LightGBM
# Identify object columns and bool columns
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'bool':
        X[col] = X[col].astype('category')

# --------------------------------------------
# 2. Stratified Cross-Validation
# --------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros(len(X))
importances = pd.DataFrame(index=X.columns)
importances['total_gain'] = 0

print("\nStarting LightGBM Training (5-Fold CV)...")

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Calculate scale_pos_weight for this fold
    # LightGBM parameter: scale_pos_weight
    # Ratio of negative / positive
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_weight = neg_count / pos_count
    
    # Define Model
    # Using small learning rate and high num_leaves for potential
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    
    # Train
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Predict
    val_preds = clf.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    # Valid AUC
    fold_auc = roc_auc_score(y_val, val_preds)
    print(f"Fold {fold} AUC: {fold_auc:.4f}")
    
    # Feature Importance
    importances['total_gain'] += clf.booster_.feature_importance(importance_type='gain')

# --------------------------------------------
# 3. Overall Evaluation
# --------------------------------------------
print("\n--- Overall Performance ---")

total_auc = roc_auc_score(y, oof_preds)
print(f"OOF ROC-AUC: {total_auc:.4f}")

# Find Optimal Threshold
precision, recall, thresholds = precision_recall_curve(y, oof_preds)
f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"Max F1 Score: {f1_scores[best_idx]:.4f}")

y_pred_optimal = (oof_preds >= best_threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y, y_pred_optimal))

print("Confusion Matrix:")
cm = confusion_matrix(y, y_pred_optimal)
print(cm)

# --------------------------------------------
# 4. Feature Importance Analysis
# --------------------------------------------
importances['total_gain'] /= 5  # Average gain
importances = importances.sort_values(by='total_gain', ascending=False)

print("\n--- Top 10 Features (Avg Gain) ---")
print(importances.head(10))

# --------------------------------------------
# 5. Save Final Model (Retrained on Full Data)
# --------------------------------------------
print("\nRetraining on full dataset for deployment...")
final_clf = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    scale_pos_weight=scale_weight, # Using last fold's weight approximation is fine, or recalc
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

final_clf.fit(X, y)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"optimized_lgbm_model_{timestamp}.joblib"
joblib.dump(final_clf, model_filename)

print(f"Model saved to: {model_filename}")
print("🚀 Pipeline Complete.")
