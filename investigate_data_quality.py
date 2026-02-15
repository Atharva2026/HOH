
import pandas as pd
import numpy as np
from data_processor import get_processed_data

# Load Data
X, y = get_processed_data("data.csv")

# 1. Class Overlap Analysis
print("\n--- Class Overlap Analysis ---")
# Compare means of top features for both classes
features = ['missed_payments_last_6m', 'failed_autodebit_count', 'risk_momentum_score']

for f in features:
    print(f"\nFeature: {f}")
    print(X.groupby(y)[f].describe()[['mean', 'std', 'min', 'max']])

# 2. Check for "Impossible" defaulters (Defaulters with perfect records)
perfect_record = (X['missed_payments_last_6m'] == 0) & \
                 (X['failed_autodebit_count'] == 0) & \
                 (X['credit_score'] > 700)

print("\n--- 'Perfect Record' Defaults ---")
print("Count of people with perfect records who still defaulted:")
print(y[perfect_record].value_counts())

# 3. Check for "Impossible" non-defaulters (Terrible records but didn't default)
terrible_record = (X['missed_payments_last_6m'] > 2) & \
                  (X['failed_autodebit_count'] > 2)

print("\n--- 'Terrible Record' Non-Defaults ---")
print("Count of people with terrible records who did NOT default:")
print(y[terrible_record].value_counts())
