
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# Set random seed
RANDOM_STATE = 42

# ==========================================
# 1. Data Cleaning (from data_cleaner.py)
# ==========================================
def clean_data(df, target_col="default_next_30_days"):
    """
    Filters out contradictory samples from the dataset.
    Robust to missing columns (works for both old and new datasets).
    """
    if target_col not in df.columns:
        return df
        
    initial_count = len(df)
    
    # Dynamic Cleaning Criteria based on available columns
    contradiction_mask = pd.Series(False, index=df.index)
    
    # Criteria 1: Perfect Record (Old Dataset)
    if 'missed_payments_last_6m' in df.columns and 'failed_autodebit_count' in df.columns and 'credit_score' in df.columns:
        contradiction_mask = (
            (df[target_col] == 1) & 
            (df['missed_payments_last_6m'] == 0) & 
            (df['failed_autodebit_count'] == 0) & 
            (df['credit_score'] > 700)
        )
    
    # Criteria 2: High Liquidity Default (New Dataset)
    # If Liquidity Ratio > 2.0 (Healthy) AND Unified Stress < 0.2 (Low Stress) but Defaulted
    elif 'liquidity_ratio' in df.columns and 'unified_stress_index' in df.columns:
        contradiction_mask = (
            (df[target_col] == 1) &
            (df['liquidity_ratio'] > 2.0) &
            (df['unified_stress_index'] < 0.2)
        )
    
    df_cleaned = df[~contradiction_mask].copy()
    removed_count = contradiction_mask.sum()
    
    print(f"\n--- Data Cleaning Report ---")
    print(f"Initial Rows: {initial_count}")
    print(f"Contradictory Samples Removed: {removed_count} ({removed_count/initial_count:.2%})")
    print(f"Final Rows: {len(df_cleaned)}")
    
    return df_cleaned

# ==========================================
# 2. Feature Engineering (from data_processor.py)
# ==========================================
class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced Feature Engineering Transformer for Credit Risk.
    Encapsulates all feature logic to ensure consistency between training and inference.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # --- NEW DATASET FEATURES ---
        if 'unified_stress_index' in df.columns and 'stress_momentum_score' in df.columns:
            # 1. Stress Acceleration
            df['stress_acceleration'] = df['unified_stress_index'] * df['stress_momentum_score']
            
            # 2. Behavioral Shock Impact
            if 'behavioral_shock_index' in df.columns:
                df['shock_amplification'] = df['behavioral_shock_index'] * (1 + df['stress_momentum_score'])
                
            # 3. Income to Spending Strain
            if 'monthly_income' in df.columns and 'total_spend' in df.columns:
                df['income_utilization'] = df['total_spend'] / (df['monthly_income'] + 1e-5)
                
            # 4. Gambling Ratio
            if 'gambling_spend' in df.columns and 'monthly_income' in df.columns:
                df['gambling_ratio'] = df['gambling_spend'] / (df['monthly_income'] + 1e-5)
            
            # 5. Salary Delay Impact
            if 'salary_delay_days' in df.columns and 'liquidity_ratio' in df.columns:
                # If salary is delayed and liquidity is low, high risk
                df['delayed_liquidity_crunch'] = df['salary_delay_days'] / (df['liquidity_ratio'] + 0.1)

        # --- OLD DATASET FEATURES (Conditional) ---
        
        # Gambling interaction (Old)
        if 'gambling_spend_increase_pct' in df.columns and 'emi_to_income_ratio' in df.columns:
            df['gambling_stress_interaction'] = df['gambling_spend_increase_pct'] * df['emi_to_income_ratio']
        
        # Panic Spending (Old)
        if 'discretionary_spend_drop_pct' in df.columns:
            df['panic_cutback_flag'] = (df['discretionary_spend_drop_pct'] > 0.3).astype(int)
        
        # Savings Erosion (Combined)
        if 'savings_decline_pct' in df.columns:
            if 'income_drop_pct' in df.columns:
                df['savings_cushion_erosion'] = df['savings_decline_pct'] * df['income_drop_pct']
            elif 'monthly_income' in df.columns: # Proxy for new dataset
                 df['savings_income_strain'] = df['savings_decline_pct'] * np.log1p(df['monthly_income'])

        # Utilization Headroom (Combined)
        if 'credit_utilization_ratio' in df.columns:
            df['utilization_headroom'] = (1 - df['credit_utilization_ratio']).clip(lower=0)
        
        # Payment History (Old)
        if 'missed_payments_last_6m' in df.columns and 'recent_failed_autodebit_flag' in df.columns:
            df['recent_delinquency_intensity'] = df['missed_payments_last_6m'] * df['recent_failed_autodebit_flag']
        
        # Total Delinquency (Combined Logic)
        if 'failed_autodebit_count' in df.columns:
             if 'missed_payments_last_6m' in df.columns:
                df['total_delinquency_events'] = df['missed_payments_last_6m'] + df['failed_autodebit_count']
             else:
                df['total_delinquency_events'] = df['failed_autodebit_count'] # Fallback for new dataset

        # Ratios & Interactions (Round 2 Logic adapted)
        if 'stress_momentum_score' in df.columns and 'financial_flexibility_score' in df.columns: # Hypothetical flexibility check
             pass 

        # Clip Outliers
        clip_columns = ['unified_stress_index', 'gambling_spend', 'total_spend', 'income_drop_pct']
        for col in clip_columns:
             if col in df.columns:
                 limit = df[col].quantile(0.99)
                 df[col] = df[col].clip(upper=limit)

        return df

def get_complete_data(filepath="financial_stress_timeseries.csv"):
    """Validates loading, cleaning, and processing."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None, None
    
    # Auto-detect target column
    if "default_next_30_days" in df.columns:
        target_col = "default_next_30_days"
    elif "default_next_4_weeks" in df.columns:
        target_col = "default_next_4_weeks"
    else:
        print("Error: No valid target column found (looked for 'default_next_30_days' or 'default_next_4_weeks').")
        return None, None
    
    print(f"Detected Target: {target_col}")

    # Clean
    df_clean = clean_data(df, target_col)
    
    y = df_clean[target_col]
    X_raw = df_clean.drop(target_col, axis=1)
    
    # Drop IDs if present
    drop_cols = ['customer_id', 'week_number', 'expected_salary_day', 'actual_salary_day']
    X_raw = X_raw.drop([c for c in drop_cols if c in X_raw.columns], axis=1)
    
    # Process
    engineer = CreditRiskFeatureEngineer()
    X_processed = engineer.transform(X_raw)
    
    return X_processed, y

# ==========================================
# 3. Model Training & Stacking
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    print("Loading and Engineering Features...")
    X, y = get_complete_data("financial_stress_timeseries.csv")
    print(f"Data Shape: {X.shape}")

    # Handle categoricals
    # Explicitly find object/string columns and convert to category
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = X[col].astype('category')
        
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'bool':
            X[col] = X[col].astype('category')

    cat_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype.name == 'category']
    cat_features_names = [col for col in X.columns if X[col].dtype.name == 'category']
    
    print(f"Categorical Features: {cat_features_names}")

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
        'random_state': RANDOM_STATE, 'n_jobs': -1, 
        # early_stopping_rounds removed for final training 
    }

    # CatBoost (Categorical Specialist)
    cb_params = {
        'iterations': 1000, 'learning_rate': 0.03, 'depth': 6,
        'loss_function': 'Logloss', 'eval_metric': 'AUC', 'scale_pos_weight': 2.5,
        'random_seed': RANDOM_STATE, 'verbose': 0,
        # early_stopping_rounds removed for final training
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
        # Use a copy of params to add early_stopping only for CV loop
        xgb_cv_params = xgb_params.copy()
        xgb_cv_params['early_stopping_rounds'] = 50
        clf_xgb = XGBClassifier(**xgb_cv_params)
        clf_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = clf_xgb.predict_proba(X_val)[:, 1]
        
        # CatBoost
        cb_cv_params = cb_params.copy()
        cb_cv_params['early_stopping_rounds'] = 50
        clf_cat = CatBoostClassifier(**cb_cv_params)
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

    # Meta CV
    meta_oof = np.zeros(len(X))
    
    for train_idx, val_idx in cv.split(X_level2, y):
        X_m_train, X_m_val = X_level2.iloc[train_idx], X_level2.iloc[val_idx]
        y_m_train, y_m_val = y.iloc[train_idx], y.iloc[val_idx]
        
        meta_clf = LogisticRegression()
        meta_clf.fit(X_m_train, y_m_train)
        
        meta_oof[val_idx] = meta_clf.predict_proba(X_m_val)[:, 1]

    final_auc = roc_auc_score(y, meta_oof)
    print(f"\n🏆 Stacking Ensemble ROC-AUC: {final_auc:.4f}")

    # Calculate and Print Detailed Metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Use 0.5 threshold for default classification
    y_pred = (meta_oof > 0.5).astype(int)
    
    print("\n--- Classification Report (Threshold = 0.5) ---")
    print(classification_report(y, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # Train final meta learner on all data
    print("\nRetraining Final Stack on Full Data...")
    
    # 1. Fit Base Models
    final_lgb = lgb.LGBMClassifier(**lgb_params)
    final_lgb.fit(X, y)
    
    final_xgb = XGBClassifier(**xgb_params)
    final_xgb.fit(X, y)
    
    final_cat = CatBoostClassifier(**cb_params)
    final_cat.fit(X, y, cat_features=cat_features_names)
    
    # 2. Fit Meta Learner
    final_meta = LogisticRegression()
    final_meta.fit(X_level2, y)
    
    print("\n✅ Training Complete. This script contains the full pipeline.")

    # 3. Save Models (Added for Dashboard)
    print("\nSaving Models for Dashboard...")
    joblib.dump(final_lgb, 'final_lgb_dashboard.joblib')
    
    # Save XGBoost using native JSON format to avoid pickling specific version errors
    final_xgb.save_model('final_xgb_dashboard.json')
    
    joblib.dump(final_cat, 'final_cat_dashboard.joblib')
    joblib.dump(final_meta, 'final_meta_dashboard.joblib')
    print("✅ Models saved as *_dashboard.joblib (and .json for XGB)")
