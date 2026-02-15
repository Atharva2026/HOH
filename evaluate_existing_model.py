import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_recall_curve

# Load Data
try:
    df = pd.read_csv("data.csv")
    print(f"Dataset Loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'data.csv' not found.")
    exit()

# Preprocessing (Must match training script exactly)
target_col = "default_next_30_days"
if target_col not in df.columns:
    print(f"Error: Target '{target_col}' not found.")
    exit()

X = df.drop(target_col, axis=1)
y = df[target_col]

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Train-Test Split (Must match training script exactly)
# random_state=42, test_size=0.2, stratify=y
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load Model
model_filename = "pred_delinquency_model_20260212_213428.joblib"
try:
    ensemble_model = joblib.load(model_filename)
    print(f"Model '{model_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found.")
    exit()

# Predict
print("Predicting on test set...")
y_prob = ensemble_model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print(f"Ensemble Test ROC-AUC: {roc_auc:.4f}")

# Find Optimal Threshold for F1-score (to match original script logic)
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * recall_curve * precision_curve / (recall_curve + precision_curve + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal Threshold (maximizing F1): {best_threshold:.4f}")

# Apply Threshold
y_pred_optimal = (y_prob >= best_threshold).astype(int)

# Calculate Recall
recall = recall_score(y_test, y_pred_optimal)
print(f"Recall: {recall:.4f}")
