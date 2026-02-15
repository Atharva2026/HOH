
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
        # Create a copy to avoid SettingWithCopyWarning
        df = X.copy()
        
        # ---------------------------------------------------------
        # 1. Behavioral Interactions (Spending & Lifestyle)
        # ---------------------------------------------------------
        
        # Gambling relative to income proxy (EMI + Rent + Living expenses proxy)
        # Higher gambling share is risky.
        df['gambling_stress_interaction'] = df['gambling_spend_increase_pct'] * df['emi_to_income_ratio']
        
        # Panic Spending Drop: Sudden cutbacks often precede default
        # Flag if discretionary spend drops by > 30%
        df['panic_cutback_flag'] = (df['discretionary_spend_drop_pct'] > 0.3).astype(int)
        
        # ---------------------------------------------------------
        # 2. Financial Resilience (Can they absorb a shock?)
        # ---------------------------------------------------------
        
        # Savings Cushion Erosion: Drop in savings + Income shock
        # This captures the "burning the candle at both ends" scenario
        df['savings_cushion_erosion'] = df['savings_decline_pct'] * df['income_drop_pct']
        
        # Utilization Headroom: How much credit line is left?
        # 1 - Utilization. If negative (over limit), clip to 0 for this feature.
        df['utilization_headroom'] = (1 - df['credit_utilization_ratio']).clip(lower=0)
        
        # ---------------------------------------------------------
        # 3. Payment History Compounders (Frequency * Recency)
        # ---------------------------------------------------------
        
        # If they missed payments AND have recent failed debits, risk compounds
        df['recent_delinquency_intensity'] = df['missed_payments_last_6m'] * df['recent_failed_autodebit_flag']
        
        # Binning Utility Payment Delays (Non-linear risk)
        # 0 days = Low, 1-30 = Med, 30+ = High Risk
        # We'll use a continuous/ordinal mapping for tree models
        df['utility_delay_severity'] = pd.cut(
            df['utility_payment_delay_days'], 
            bins=[-1, 0, 15, 30, 1000], 
            labels=[0, 1, 3, 5]
        ).astype(int)
        
        # Total Delinquency Events (Simple but effective sum)
        df['total_delinquency_events'] = df['missed_payments_last_6m'] + df['failed_autodebit_count']
        
        # ---------------------------------------------------------
        # NEW FEATURES (Round 2)
        # ---------------------------------------------------------
        # 1. Interaction with Credit Score (Inverse Scaling)
        # Normalizing delinquencies by credit tier (higher score should mean lower tolerance for errors)
        for col in ['failed_autodebit_count', 'utility_payment_delay_days', 'income_drop_pct']:
            df[f'{col}_per_credit_tier'] = df[col] / (df['credit_score'] / 100 + 1e-5)
            
        # 2. Stress Concentration (Ratio of Ratios)
        # High EMI burden with low headroom = extreme stress
        # Utilization headroom was not explicitly created before, let's proxy it
        # Utilization is 0-1, so headroom is 1 - utilization
        df['utilization_headroom'] = (1 - df['credit_utilization_ratio']).clip(lower=0.01)
        df['stress_concentration'] = df['emi_to_income_ratio'] / df['utilization_headroom']
        
        # 3. Perfect Storm (Boolean Interaction)
        # Combination of Income Shock + Savings Erosion + Delinquency
        df['perfect_storm'] = (
            (df['income_drop_pct'] > 0.15) & 
            (df['savings_decline_pct'] > 0.3) & 
            (df['missed_payments_last_6m'] > 0)
        ).astype(int)
        
        # ---------------------------------------------------------
        # 4. Synthesized Ratios & Stress Tests
        # ---------------------------------------------------------
        
        # Adjusted Obligation Ratio based on Savings Trend
        # If savings are declining, the effective burden of EMI is higher
        # Logic: EMI_Ratio * (1 + Savings_Decline)
        df['adjusted_obligation_burden'] = df['emi_to_income_ratio'] * (1 + df['savings_decline_pct'].clip(lower=0))

        # Risk Velocity: Momentum normalized by Credit Score (Inverse relationship)
        # Adding epsilon to avoid div by zero
        df['risk_velocity'] = df['risk_momentum_score'] / (df['credit_score'] + 1e-5)
        
        # ---------------------------------------------------------
        # 5. Trend Analysis (Volatile Decline)
        # ---------------------------------------------------------
        # High volatility + Declining savings = Loss of control
        df['volatility_risk_factor'] = df['balance_volatility_30d'] * df['savings_decline_pct']

        # ---------------------------------------------------------
        # 6. Outlier Handling (Clipping extreme values for stability)
        # ---------------------------------------------------------
        # Tree models are robust, but extreme outliers can still skew splits or affect regularization
        clip_columns = ['income_drop_pct', 'savings_decline_pct', 'gambling_spend_increase_pct']
        for col in clip_columns:
             if col in df.columns:
                 # Clip at 99th percentile (approx. 3 std devs usually)
                 limit = df[col].quantile(0.99)
                 df[col] = df[col].clip(upper=limit)

        return df

def get_processed_data(filepath="data.csv"):
    """
    Helper function to load and process data in one go.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None, None

    target_col = "default_next_30_days"
    if target_col not in df.columns:
        print(f"Error: Target '{target_col}' not found.")
        return None, None
        
    y = df[target_col]
    X_raw = df.drop(target_col, axis=1)
    
    # Process Features
    engineer = CreditRiskFeatureEngineer()
    X_processed = engineer.transform(X_raw)
    
    return X_processed, y
