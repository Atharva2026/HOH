import pandas as pd
import numpy as np

# Load Data
try:
    df = pd.read_csv("data.csv")
    print(f"Dataset Loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'data.csv' not found.")
    exit()

target_col = "default_next_30_days"

# 1. Feature Correlations with Target
print("\n--- Feature Correlations with Target ---")
corr = df.corr()[target_col].sort_values(ascending=False)
print(corr)

# 2. Distribution of potential high-impact features by class
high_impact_features = [
    'credit_score', 'emi_to_income_ratio', 'credit_utilization_ratio',
    'missed_payments_last_6m', 'savings_decline_pct', 'risk_momentum_score'
]

print("\n--- Feature Means by Target Class ---")
print(df.groupby(target_col)[high_impact_features].mean())

# 3. Check for specific interaction candidates
# "Stress" check: High utilization AND High EMI
df['high_stress'] = ((df['credit_utilization_ratio'] > 0.7) & (df['emi_to_income_ratio'] > 0.4)).astype(int)
print("\n--- 'High Stress' (High Util + High EMI) vs Target ---")
print(df.groupby('high_stress')[target_col].mean())
print(f"Count of High Stress cases: {df['high_stress'].sum()}")

# "Trend" check: Savings declining AND Volatile balance
df['volatile_decline'] = ((df['savings_decline_pct'] > 0.1) & (df['balance_volatility_30d'] > 0.2)).astype(int)
print("\n--- 'Volatile Decline' (Savings Drop + Volatility) vs Target ---")
print(df.groupby('volatile_decline')[target_col].mean())
print(f"Count of Volatile Decline cases: {df['volatile_decline'].sum()}")
