# ============================================
# Advanced Pre-Delinquency ML Model
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, 
    precision_recall_curve, f1_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# Set random seed for reproducibility
RANDOM_STATE = 42

# --------------------------------------------
# 1. Load Data
# --------------------------------------------
try:
    df = pd.read_csv("data.csv")
    print(f"Dataset Loaded Successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please ensure the file exists.")
    exit()

# Basic Data Inspection
print(df.head())

# --------------------------------------------
# 2. Preprocessing & Feature Engineering
# --------------------------------------------
# Handle missing values broadly (though pipelines handle this better, doing a quick check helps)
# Verify if target column exists
target_col = "default_next_30_days"
if target_col not in df.columns:
    print(f"Error: Target column '{target_col}' not found.")
    exit()

X = df.drop(target_col, axis=1)
y = df[target_col]

# Check for categorical columns and encode if necessary
# For this advanced script, we'll use One-Hot Encoding for flexibility if strings exist
X = pd.get_dummies(X, drop_first=True)

# --------------------------------------------
# 3. Train-Test Split (Stratified)
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# Calculate scale_pos_weight for XGBoost (handling class imbalance)
negative_count = len(y_train) - y_train.sum()
positive_count = y_train.sum()
scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1

print(f"Class Imbalance Ratio (scale_pos_weight): {scale_pos_weight:.2f}")

# --------------------------------------------
# 4. Model Pipeline Definition
# --------------------------------------------

# Define Preprocessing Pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values robustly
    ('scaler', StandardScaler())  # Scale features (helpful for some models, neutral for trees)
])

# Define XGBoost Classifier
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Define Random Forest Classifier (as an ensemble partner)
rf_clf = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# --------------------------------------------
# 5. Hyperparameter Tuning (RandomizedSearchCV)
# --------------------------------------------
print("\nStarting Hyperparameter Tuning for XGBoost...")

xgb_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5]
}

# Use StratifiedKFold for robust cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

xgb_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=xgb_param_dist,
    n_iter=20,  # Number of parameter settings that are sampled
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Fit the search - Imputing within the search loop prevents data leakage,
# but since tree models handle missing values well, we can just fit X_train directly
# or use the preprocessor. Let's use the raw X_train for XGBoost to leverage its native handling
# unless there are NaNs that cause issues.
# For simplicity and power, we will fit directly on X_train (XGBoost handles NaNs).
xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_
print(f"Best XGBoost Params: {xgb_search.best_params_}")
print(f"Best CV ROC-AUC: {xgb_search.best_score_:.4f}")

# --------------------------------------------
# 6. Ensemble Model (Voting Classifier)
# --------------------------------------------
# We combine the tuned XGBoost with a Random Forest for stability
print("\nTraining Ensemble Model (XGBoost + Random Forest)...")

ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', rf_clf)
    ],
    voting='soft',
    n_jobs=-1
)

# Fit Ensemble
ensemble_model.fit(X_train, y_train)

# --------------------------------------------
# 7. Model Evaluation & Threshold Tuning
# --------------------------------------------
print("\nEvaluating Model...")

# Get probability predictions
y_prob = ensemble_model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print(f"Ensemble Test ROC-AUC: {roc_auc:.4f}")

# Find Optimal Threshold for F1-score
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal Threshold (maximizing F1): {best_threshold:.4f}")

# Apply Threshold
y_pred_optimal = (y_prob >= best_threshold).astype(int)

print("\nClassification Report (Optimal Threshold):")
print(classification_report(y_test, y_pred_optimal))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_optimal)
print(cm)

# --------------------------------------------
# 8. Feature Importance (from XGBoost)
# --------------------------------------------
# Since Ensemble doesn't have a simple 'feature_importances_', we show XGBoost's
importance = best_xgb.feature_importances_
features = X.columns

# Sort features
indices = np.argsort(importance)[-15:]  # Top 15 features
plt.figure(figsize=(10, 8))
plt.title("Top 15 Feature Importances (XGBoost)")
plt.barh(range(len(indices)), importance[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 9. SHAP Explainability
# --------------------------------------------
# SHAP works best with the base tree model
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

print("\nGenerating SHAP Summary Plot...")
try:
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('shap_summary.png', bbox_inches='tight')
    plt.show()
    print("SHAP plot saved as 'shap_summary.png'")
except Exception as e:
    print(f"Could not generate SHAP plot: {e}")

# --------------------------------------------
# 10. Save Results & Model
# --------------------------------------------
# Add predictions to original dataset (simulated logic for the whole dataset)
df_scored = df.copy()
# Note: For production, you should re-process the whole df exactly as X_train.
# Here we'll just predict on the whole X, assuming it fits memory.
# We need to ensure X has same columns (it does, as we just dropped target)
full_probs = ensemble_model.predict_proba(X)[:, 1]

def risk_tier(prob):
    if prob < 0.3: return "Low"      # Adjusted thresholds based on typical imbalance
    elif prob < 0.6: return "Medium"
    elif prob < 0.8: return "High"
    else: return "Critical"

df_scored["risk_probability"] = full_probs
df_scored["risk_category"] = df_scored["risk_probability"].apply(risk_tier)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"scored_customers_{timestamp}.csv"
df_scored.to_csv(output_file, index=False)
print(f"\nScored data saved to: {output_file}")

# Save the trained model
model_filename = f"pred_delinquency_model_{timestamp}.joblib"
joblib.dump(ensemble_model, model_filename)
print(f"Trained model saved to: {model_filename}")

print("\nPipeline Completed Successfully 🚀")
