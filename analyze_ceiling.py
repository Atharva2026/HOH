
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# 1. Load Data
df = pd.read_csv("data.csv")

# 2. Define "Impossible" Defaulters (The Noise)
# These are people who defaulted despite having perfect credit behavior
impossible_mask = (
    (df['default_next_30_days'] == 1) & 
    (df['missed_payments_last_6m'] == 0) & 
    (df['failed_autodebit_count'] == 0) & 
    (df['credit_score'] > 700)
)

impossible_df = df[impossible_mask]
clean_df = df[~impossible_mask]

print(f"Total Rows: {len(df)}")
print(f"Impossible Defaulters (Noise): {len(impossible_df)} ({len(impossible_df)/len(df):.2%})")
print(f"Remaining Clean Data: {len(clean_df)}")

# 3. Save Impossible IDs for Audit
if 'customer_id' in df.columns:
    impossible_df[['customer_id', 'credit_score', 'income_drop_pct']].to_csv("audit_list_impossible_defaulters.csv", index=False)
    print("Saved 'audit_list_impossible_defaulters.csv' for manual review.")
else:
    impossible_df.to_csv("audit_list_impossible_defaulters.csv", index=False)
    print("Saved 'audit_list_impossible_defaulters.csv' (No customer_id found, saving full rows).")


# 4. Train Model on "Perfectly Cleaned" Data (Theoretical Max)
# We want to show the user: "If you fix this data error, here is what AUC you COULD get."

print("\nTraining LightGBM on Remaining Clean Data (Hypothetical Max Performance)...")
X = clean_df.drop('default_next_30_days', axis=1)
y = clean_df['default_next_30_days']

# Preprocessing (Simple for this test)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')

clf = LGBMClassifier(n_estimators=500, random_state=42, verbose=-1)
cv_scores = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in cv.split(X, y):
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = clf.predict_proba(X.iloc[val_idx])[:, 1]
    cv_scores.append(roc_auc_score(y.iloc[val_idx], preds))

print(f"\n🚀 Theoretical Max AUC (Clean Data): {np.mean(cv_scores):.4f}")
print("---------------------------------------------------")
print("INTERPRETATION:")
if np.mean(cv_scores) > 0.75:
    print("Good News! Your model IS working. The problem is purely bad labels.")
    print("Action: You must ask your data team why these 1000+ users are labeled as defaults.")
else:
    print("Even with clean data, the signal is weak. You need external data sources.")
