
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from data_cleaner import get_cleaned_processed_data
import joblib

# Set random seed
RANDOM_STATE = 42

# 1. Load Data
print("Loading and Engineering Features...")
X, y = get_cleaned_processed_data("data.csv")
print(f"Data Shape: {X.shape}")

# Handle categoricals
# XGBoost/LGBM need category dtype. CatBoost handles strings/int/category.
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'bool':
        X[col] = X[col].astype('category')

cat_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype.name == 'category']
cat_features_names = [col for col in X.columns if X[col].dtype.name == 'category']

# 2. Define Base Models
# LightGBM (Tuned)
lgb_params = {
    'n_estimators': 1351, 'learning_rate': 0.031, 'num_leaves': 40, 'max_depth': 3,
    'min_child_samples': 71, 'subsample': 0.956, 'colsample_bytree': 0.837,
    'reg_alpha': 7e-05, 'reg_lambda': 0.0008, 'scale_pos_weight': 2.475,
    'objective': 'binary', 'metric': 'auc', 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1
}

# XGBoost (Robust)
xgb_params = {
    'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 4,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': 2.5,
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'enable_categorical': True,
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'early_stopping_rounds': 50
}

# CatBoost (Categorical Specialist)
# Note: For CatBoost inside the loop, we'll pass cat_features
cb_params = {
    'iterations': 1000, 'learning_rate': 0.03, 'depth': 6,
    'loss_function': 'Logloss', 'eval_metric': 'AUC', 'scale_pos_weight': 2.5,
    'random_seed': RANDOM_STATE, 'verbose': 0, 'early_stopping_rounds': 50
}

# 3. Generate OOF Predictions (Level 1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

print("\nStarting Stacking Level 1 (Base Models)...")

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM
    clf_lgb = lgb.LGBMClassifier(**lgb_params)
    clf_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = clf_lgb.predict_proba(X_val)[:, 1]
    
    # XGBoost
    clf_xgb = XGBClassifier(**xgb_params)
    clf_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = clf_xgb.predict_proba(X_val)[:, 1]
    
    # CatBoost
    # Provide cat features explicitly
    clf_cat = CatBoostClassifier(**cb_params)
    clf_cat.fit(X_train, y_train, cat_features=cat_features_names, eval_set=(X_val, y_val))
    oof_cat[val_idx] = clf_cat.predict_proba(X_val)[:, 1]
    
    print(f"Fold {fold} Done.")

print(f"\nLGBM AUC: {roc_auc_score(y, oof_lgb):.4f}")
print(f"XGB  AUC: {roc_auc_score(y, oof_xgb):.4f}")
print(f"CAT  AUC: {roc_auc_score(y, oof_cat):.4f}")

# 4. Meta Learner (Level 2)
print("\nTraining Meta-Learner (Logistic Regression)...")
X_level2 = pd.DataFrame({
    'lgb': oof_lgb,
    'xgb': oof_xgb,
    'cat': oof_cat
})

# Use CV for Meta Learner evaluation as well to avoid overfitting
meta_oof = np.zeros(len(X))
meta_coefs = []

for train_idx, val_idx in cv.split(X_level2, y):
    X_m_train, X_m_val = X_level2.iloc[train_idx], X_level2.iloc[val_idx]
    y_m_train, y_m_val = y.iloc[train_idx], y.iloc[val_idx]
    
    meta_clf = LogisticRegression()
    meta_clf.fit(X_m_train, y_m_train)
    
    meta_oof[val_idx] = meta_clf.predict_proba(X_m_val)[:, 1]
    meta_coefs.append(meta_clf.coef_[0])

final_auc = roc_auc_score(y, meta_oof)
print(f"\n🏆 Stacking Ensemble ROC-AUC: {final_auc:.4f}")

print("\nMeta-Learner Weights (Avg):")
avg_coefs = np.mean(meta_coefs, axis=0)
print(f"LGBM: {avg_coefs[0]:.4f}")
print(f"XGB:  {avg_coefs[1]:.4f}")
print(f"CAT:  {avg_coefs[2]:.4f}")

if final_auc > 0.70:
    print("\n✅ GOAL ACHIEVED!")
else:
    print("\n⚠️ GOAL MISSED (But saving best model anyway).")

# Train final meta learner on all data
print("Retraining Final Stack on Full Data...")
# 1. Fit Base Models on Full Data
final_lgb = lgb.LGBMClassifier(**lgb_params)
final_lgb.fit(X, y)
joblib.dump(final_lgb, "final_lgbm_stack_base.joblib")

final_xgb = XGBClassifier(**{k: v for k, v in xgb_params.items() if k != 'early_stopping_rounds'})
final_xgb.fit(X, y)
joblib.dump(final_xgb, "final_xgb_stack_base.joblib")

final_cat = CatBoostClassifier(**{k: v for k, v in cb_params.items() if k != 'early_stopping_rounds'})
final_cat.fit(X, y, cat_features=cat_features_names)
final_cat.save_model("final_cat_stack_base.cbm")

# 2. Fit Meta Learner
final_meta = LogisticRegression()
final_meta.fit(X_level2, y)
joblib.dump(final_meta, "final_stacking_meta.joblib")

print("\n📦 All models saved:")
print("- final_lgbm_stack_base.joblib")
print("- final_xgb_stack_base.joblib")
print("- final_cat_stack_base.cbm")
print("- final_stacking_meta.joblib")
