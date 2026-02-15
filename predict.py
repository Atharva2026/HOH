
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from data_cleaner import get_cleaned_processed_data 
# Note: In production you'd use data_processor directly on new data, 
# but for this example we Reuse the cleaning/processing logic if needed.
from data_processor import CreditRiskFeatureEngineer

class StackingPredictor:
    def __init__(self):
        self.lgb = joblib.load("final_lgbm_stack_base.joblib")
        self.xgb = joblib.load("final_xgb_stack_base.joblib")
        self.cat = CatBoostClassifier()
        self.cat.load_model("final_cat_stack_base.cbm")
        self.meta = joblib.load("final_stacking_meta.joblib")
        self.engineer = CreditRiskFeatureEngineer()

    def predict_proba(self, X_raw_df):
        # 1. Engineer Features
        X_processed = self.engineer.transform(X_raw_df)
        
        # 2. Handle Categoricals
        # Each model needs its specific format.
        # LGBM/XGB need category, CatBoost needs strings/int
        
        X_lgb = X_processed.copy()
        X_xgb = X_processed.copy()
        X_cat = X_processed.copy()
        
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object' or X_processed[col].dtype == 'bool':
                X_lgb[col] = X_lgb[col].astype('category')
                X_xgb[col] = X_xgb[col].astype('category')
                X_cat[col] = X_cat[col].astype(str)
                
        # 3. Base Predictions
        p_lgb = self.lgb.predict_proba(X_lgb)[:, 1]
        p_xgb = self.xgb.predict_proba(X_xgb)[:, 1]
        # CatBoost predict_proba expects specific format
        p_cat = self.cat.predict_proba(X_cat)[:, 1]
        
        # 4. Meta Prediction
        X_meta = pd.DataFrame({'lgb': p_lgb, 'xgb': p_xgb, 'cat': p_cat})
        final_prob = self.meta.predict_proba(X_meta)[:, 1]
        
        return final_prob

if __name__ == "__main__":
    print("Loading Stacking Model...")
    model = StackingPredictor()
    
    print("Loading Sample Data...")
    df = pd.read_csv("data.csv").head(5) # Simulate new data
    if "default_next_30_days" in df.columns:
        df = df.drop("default_next_30_days", axis=1)
    
    print("Predicting...")
    probs = model.predict_proba(df)
    
    print("\nPredictions:")
    for i, p in enumerate(probs):
        print(f"Customer {i}: Risk {p:.4f}")
