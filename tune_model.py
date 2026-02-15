
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from data_cleaner import get_cleaned_processed_data

# Set random seed
RANDOM_STATE = 42

def objective(trial):
    # Load Data (This is fast since we already processed it, but for strictness we could load once outside)
    # To avoid loading every time, we'll use the global X, y loaded below
    
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)
    }

    # 5-Fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create dataset for LightGBM
        # LightGBM sklearn API is easier for simple fit/predict
        clf = lgb.LGBMClassifier(**param)
        
        # Add early stopping callback separately
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=callbacks
        )
        
        preds = clf.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        cv_scores.append(score)
        
    return np.mean(cv_scores)

if __name__ == "__main__":
    print("Loading Data for Optimization...")
    X, y = get_cleaned_processed_data("data.csv")
    
    # Identify categoricals
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'bool':
            X[col] = X[col].astype('category')
            
    print(f"Data Loaded: {X.shape}")
    
    study = optuna.create_study(direction='maximize')
    print("Starting Optimization (30 trials)...")
    study.optimize(objective, n_trials=30)
    
    print("\n--- Optimization Finished ---")
    print(f"Best ROC-AUC: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    # Train Final Model with Best Params
    print("\nTraining Final Model with Best Parameters...")
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1
    
    final_clf = lgb.LGBMClassifier(**best_params)
    final_clf.fit(X, y)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"tuned_lgbm_model_{timestamp}.joblib"
    joblib.dump(final_clf, model_filename)
    print(f"Tuned Model Saved to: {model_filename}")
