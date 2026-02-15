
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from data_processor import get_processed_data

# Set random seed
RANDOM_STATE = 42

# --------------------------------------------
# 1. Load & Process Data
# --------------------------------------------
print("Loading and Engineering Features...")
X, y = get_processed_data("data.csv")

# --------------------------------------------
# 2. Strict Feature Selection
# --------------------------------------------
# Based on the previous run's importance, we keep only the heavy hitters.
# We drop everything else to reduce noise.
selected_features = [
    'missed_payments_last_6m',
    'failed_autodebit_count',
    'risk_momentum_score',
    'credit_utilization_ratio',
    'adjusted_obligation_burden',  # Our new feature worked!
    'gambling_spend_increase_pct'
]

print(f"Selecting Top {len(selected_features)} Features: {selected_features}")
X_selected = X[selected_features]

# --------------------------------------------
# 3. Simplified Training (Regularized)
# --------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(X))

print("\nStarting Simplified Training (5-Fold CV)...")

for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y), 1):
    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Simple Weighting (No extreme imbalance handling)
    # Just standard balanced weights
    scale_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=15,       # Very restricted
        max_depth=4,         # Very shallow
        min_child_samples=50,# High requirement for split
        reg_alpha=1.0,       # L1 Regularization
        reg_lambda=1.0,      # L2 Regularization
        scale_pos_weight=scale_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    val_preds = clf.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, val_preds):.4f}")

total_auc = roc_auc_score(y, oof_preds)
print(f"\n--- Simplified Model OOF ROC-AUC: {total_auc:.4f} ---")
