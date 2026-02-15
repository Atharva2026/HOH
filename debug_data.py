
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("data.csv")
target_col = "default_next_30_days"

print(f"Data Shape: {df.shape}")
print(f"Target Distribution:\n{df[target_col].value_counts(normalize=True)}")

# 1. Check for infinite/NaN values in raw data
print("\nMissing Values in Raw Data:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 2. Re-create the engineered features to check for potential overflows
X = df.drop(target_col, axis=1)
# Simulate the engineering from improved_model_v2.py
X['debt_stress_index'] = X['emi_to_income_ratio'] * X['credit_utilization_ratio']
X['income_stability_shock'] = X['salary_delay_days'] * X['income_drop_pct']
X['total_delinquency_events'] = X['missed_payments_last_6m'] + X['failed_autodebit_count']
X['financial_retrenchment'] = X['discretionary_spend_drop_pct'] * X['savings_decline_pct']
X['recent_payment_trouble'] = X['recent_failed_autodebit_flag'] * X['utility_payment_delay_days']
X['risk_velocity'] = X['risk_momentum_score'] / (X['credit_score'] + 1e-5)
X['volatility_impact'] = X['balance_volatility_30d'] * X['savings_decline_trend']

print("\n--- Checking Engineered Features for extreme values ---")
engineered_cols = ['debt_stress_index', 'income_stability_shock', 'total_delinquency_events', 
                   'financial_retrenchment', 'recent_payment_trouble', 'risk_velocity', 'volatility_impact']

for col in engineered_cols:
    print(f"\nFeature: {col}")
    print(f"  Min: {X[col].min()}, Max: {X[col].max()}")
    print(f"  NaNs: {X[col].isna().sum()}")
    print(f"  Infs: {np.isinf(X[col]).sum()}")

# 3. Correlation with Target
# Correlation requires numeric data, let's drop categorical for a second or get dummies
X_numeric = X.select_dtypes(include=[np.number])
X_numeric[target_col] = df[target_col]

corr = X_numeric.corr()[target_col].sort_values(ascending=False)
print("\n--- Top Correlations with Target ---")
print(corr)

# 4. Check 'risk_momentum_score' and 'credit_score' specifically as they caused the division
print("\nCheck Credit Score stats:")
print(df['credit_score'].describe())
