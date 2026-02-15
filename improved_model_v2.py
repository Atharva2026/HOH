
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, 
    precision_recall_curve, f1_score, accuracy_score, recall_score
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Set random seed
RANDOM_STATE = 42

# --------------------------------------------
# 1. Load Data
# --------------------------------------------
try:
    df = pd.read_csv("data.csv")
    print(f"Dataset Loaded Successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'data.csv' not found.")
    exit()

# --------------------------------------------
# 2. Advanced Feature Engineering
# --------------------------------------------
def engineer_features(data):
    # Copy to avoid SettingWithCopy warnings
    X = data.copy()
    
    # 1. Financial Stress Indicators
    # combine debt burden ratios
    X['debt_stress_index'] = X['emi_to_income_ratio'] * X['credit_utilization_ratio']
    
    # 2. Income Stability
    # combine delayed salary with income drops
    X['income_stability_shock'] = X['salary_delay_days'] * X['income_drop_pct']
    
    # 3. Delinquency History
    # combine missed payments and failed debits
    X['total_delinquency_events'] = X['missed_payments_last_6m'] + X['failed_autodebit_count']
    
    # 4. Expenditure Cutbacks (Sign of distress)
    # combine discretionary spending reduction with savings decline
    X['financial_retrenchment'] = X['discretionary_spend_drop_pct'] * X['savings_decline_pct']
    
    # 5. Recent Payment Issues
    # flag recent issues specifically
    X['recent_payment_trouble'] = X['recent_failed_autodebit_flag'] * X['utility_payment_delay_days']
    
    # 6. Risk Momentum Interaction
    # how fast risk is increasing relative to credit score (inverse relationship expected)
    # Adding small epsilon to avoid division by zero
    X['risk_velocity'] = X['risk_momentum_score'] / (X['credit_score'] + 1e-5)
    
    # 7. Balance Volatility Impact
    # High volatility with low savings trend is dangerous
    X['volatility_impact'] = X['balance_volatility_30d'] * X['savings_decline_trend']
    
    return X

# Apply feature engineering
target_col = "default_next_30_days"
if target_col not in df.columns:
    print(f"Error: Target '{target_col}' not found.")
    exit()

# Separate potential target
y = df[target_col]
X_raw = df.drop(target_col, axis=1)

# Engineer features
X_eng = engineer_features(X_raw)

# One-Hot Encoding for categorical features
X = pd.get_dummies(X_eng, drop_first=True)

# Check for infinite values created by division (e.g. risk_velocity) and replace
X.replace([np.inf, -np.inf], np.nan, inplace=True)
# Fill resultant NaNs with 0 or median (simple strategy)
X.fillna(0, inplace=True)  # Simple fill for engineered features

print(f"Feature Engineering Complete. New Shape: {X.shape}")
print(f"New Features Added: {[col for col in X.columns if col not in X_raw.columns]}")

# --------------------------------------------
# 3. Train-Test Split
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# Calculate scale_pos_weight
neg_count = len(y_train) - y_train.sum()
pos_count = y_train.sum()
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
print(f"Class Imbalance Ratio: {scale_pos_weight:.2f}")

# --------------------------------------------
# 4. Model Definitions & Pipelines
# --------------------------------------------

# Common Preprocessor
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# -- Model 1: XGBoost (Tree-based, handles non-linearities well) --
# Note: XGBoost has its own internal handling of missing values, but standardization helps convergence sometimes
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    n_estimators=300,        # Increased
    learning_rate=0.05,      # Lower learning rate
    max_depth=6,             # Moderate depth
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1
)

# -- Model 2: Gradient Boosting Classifier (sklearn) --
gb_clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=RANDOM_STATE,
    validation_fraction=0.1,
    n_iter_no_change=10
)

# -- Model 3: Random Forest (Bagging, robust to noise) --
rf_clf = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    max_depth=10,            # Constrain depth to prevent overfitting
    min_samples_leaf=4,      # Regularization
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# -- Model 4: Extra Trees (More randomness, reduces variance) --
et_clf = ExtraTreesClassifier(
    n_estimators=300,
    class_weight='balanced',
    max_depth=10,
    min_samples_leaf=4,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# -- Model 5: Logistic Regression (Linear baseline, surprisingly effective sometimes) --
lr_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1))
])


# --------------------------------------------
# 5. Ensemble Modeling (Voting)
# --------------------------------------------
print("\nTraining Advanced Ensemble Model...")

# Weighted voting: Giving slightly more weight to gradient boosting methods which often have higher AUC
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_clf),
        ('gb', gb_clf),
        ('rf', rf_clf),
        ('et', et_clf),
        ('lr', lr_clf)
    ],
    voting='soft',
    weights=[3, 2, 2, 1, 1],  # Weights based on expected performance contribution
    n_jobs=-1
)

# Use Random Search to fine-tune weights or just fit directly?
# Fitting directly to save time, given the complexity
ensemble_model.fit(X_train, y_train)

# --------------------------------------------
# 6. Evaluation
# --------------------------------------------
print("\nEvaluating Improved Model...")

# Get probability predictions
y_prob = ensemble_model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print(f"Ensemble Test ROC-AUC: {roc_auc:.4f}")

# Find Optimal Threshold for F1 and Recall Balance
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
best_f1_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_f1_idx]
best_f1 = f1_scores[best_f1_idx]

print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"Max F1 Score: {best_f1:.4f}")

# Apply Threshold
y_pred_optimal = (y_prob >= best_threshold).astype(int)

# Detailed Metrics
print("\nClassification Report (Optimal Threshold):")
print(classification_report(y_test, y_pred_optimal))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_optimal)
print(cm)

# Calculate Recall specifically
final_recall = recall_score(y_test, y_pred_optimal)
print(f"Recall at Opt Threshold: {final_recall:.4f}")

# --------------------------------------------
# 7. Save Artifacts
# --------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"improved_delinquency_model_{timestamp}.joblib"
joblib.dump(ensemble_model, model_filename)
print(f"\nImproved model saved to: {model_filename}")

# Feature Importance (Proxy using Random Forest within ensemble for simplicity)
# Note: VotingClassifier doesn't have feature_importances_ directly
# We can extract it from the fitted RF or XGB component
try:
    # Access the fitted estimators
    # The order depends on how VotingClassifier stores them. Usually 'estimators_' attribute.
    # Let's inspect one of the tree models
    fitted_rf = ensemble_model.named_estimators_['rf']
    importances = fitted_rf.feature_importances_
    indices = np.argsort(importances)[-15:]
    
    plt.figure(figsize=(10, 8))
    plt.title("Top 15 Feature Importances (from Random Forest Component)")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(f"improved_feature_importance_{timestamp}.png")
    print(f"Feature importance plot saved.")
except Exception as e:
    print(f"Could not save feature importance: {e}")

print("\nProcess Completed 🚀")
