
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from data_cleaner import get_cleaned_processed_data

# Set random seed
RANDOM_STATE = 42

# --------------------------------------------
# 1. Load & Process Data
# --------------------------------------------
print("Loading, Cleaning, and Engineering Features...")
X, y = get_cleaned_processed_data("data.csv")
print(f"Data Shape: {X.shape}")

# Handle categoricals for XGBoost (Enable categorical support or One-Hot)
# XGBoost supports categoricals but needs 'category' dtype and enable_categorical=True
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'bool':
        X[col] = X[col].astype('category')

# --------------------------------------------
# 2. Define Models
# --------------------------------------------

# Tuned LightGBM Params (from Optuna)
lgb_params = {
    'n_estimators': 1351,
    'learning_rate': 0.03118,
    'num_leaves': 40,
    'max_depth': 3,
    'min_child_samples': 71,
    'subsample': 0.9558,
    'colsample_bytree': 0.8371,
    'reg_alpha': 7e-05,
    'reg_lambda': 0.0008,
    'scale_pos_weight': 2.4752,
    'objective': 'binary',
    'metric': 'auc',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': -1
}

# Robust XGBoost Params (Standard high-performance set)
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 2.5, # Similar to LGBM
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'enable_categorical': True,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# --------------------------------------------
# 3. Training & Evaluation (5-Fold CV)
# --------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

print("\nStarting Ensemble Training (5-Fold CV)...")

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train LightGBM
    clf_lgb = lgb.LGBMClassifier(**lgb_params)
    clf_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_lgb[val_idx] = clf_lgb.predict_proba(X_val)[:, 1]
    
    # Train XGBoost
    clf_xgb = XGBClassifier(**xgb_params, early_stopping_rounds=50)
    clf_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    oof_xgb[val_idx] = clf_xgb.predict_proba(X_val)[:, 1]
    
    # Fold scores
    auc_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
    auc_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    auc_ens = roc_auc_score(y_val, (oof_lgb[val_idx] + oof_xgb[val_idx])/2)
    
    print(f"Fold {fold} AUC -> LGB: {auc_lgb:.4f} | XGB: {auc_xgb:.4f} | Ens: {auc_ens:.4f}")

# --------------------------------------------
# 4. Overall Results
# --------------------------------------------
total_auc_lgb = roc_auc_score(y, oof_lgb)
total_auc_xgb = roc_auc_score(y, oof_xgb)
oof_ens = (oof_lgb + oof_xgb) / 2
total_auc_ens = roc_auc_score(y, oof_ens)

print("\n--- Final Results ---")
print(f"LightGBM AUC: {total_auc_lgb:.4f}")
print(f"XGBoost AUC:  {total_auc_xgb:.4f}")
print(f"Ensemble AUC: {total_auc_ens:.4f}")

if total_auc_ens > 0.70:
    print("\n✅ GOAL ACHIEVED: AUC > 0.70")
else:
    print("\n⚠️ GOAL MISSED: Need further improvements.")

# --------------------------------------------
# 5. Save Artifacts
# --------------------------------------------
# Save the final Ensemble predictions for analysis
results_df = pd.DataFrame({
    'actual': y,
    'prob_lgbm': oof_lgb,
    'prob_xgb': oof_xgb,
    'prob_ensemble': oof_ens
})
results_df.to_csv("ensemble_results.csv", index=False)
print("Saved ensemble validation results to 'ensemble_results.csv'")

# Retrain on full data and save
print("Retraining final models on full data...")
final_lgb = lgb.LGBMClassifier(**lgb_params)
final_lgb.fit(X, y)
joblib.dump(final_lgb, "final_lgbm.joblib")

final_xgb = XGBClassifier(**xgb_params)
final_xgb.fit(X, y)
joblib.dump(final_xgb, "final_xgb.joblib")

print("Saved final models.")
