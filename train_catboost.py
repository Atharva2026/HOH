
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from data_cleaner import get_cleaned_processed_data

# Set random seed
RANDOM_STATE = 42

# 1. Load Data
print("Loading Data...")
X, y = get_cleaned_processed_data("data.csv")

# CatBoost handles categoricals, but SMOTE requires numeric encoding usually.
# However, CatBoost can handle raw strings.
# SMOTE needs numeric.
# We need to encode categoricals for SMOTE, then pass to CatBoost.
# Or better: use CatBoost's internal handling and avoid SMOTE if it complicates things?
# Let's try CatBoost with Class Weights first (simpler). If that fails, SMOTE.

cat_features = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype == 'category' or X[col].dtype == 'bool']
print(f"Categorical Features: {cat_features}")

# Ensure boolean are cast to int or string for CatBoost
for col in cat_features:
    X[col] = X[col].astype(str)

# 2. Training with 5-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(X))

print("\nStarting CatBoost Training...")

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Calculate scale weight
    scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    clf = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        cat_features=cat_features,
        scale_pos_weight=scale_weight,
        random_seed=RANDOM_STATE,
        verbose=200,
        early_stopping_rounds=100
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, oof_preds[val_idx]):.4f}")

total_auc = roc_auc_score(y, oof_preds)
print(f"\n--- CatBoost ROC-AUC: {total_auc:.4f} ---")

if total_auc > 0.70:
    print("✅ GOAL ACHIEVED!")
    clf.save_model("catboost_model.cbm")
else:
    print("⚠️ GOAL MISSED.")
