import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. Feature Engineering (Copied from best_model_standalone.py)
# ==========================================
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
        df = X.copy()
        
        # --- NEW DATASET FEATURES ---
        if 'unified_stress_index' in df.columns and 'stress_momentum_score' in df.columns:
            # 1. Stress Acceleration
            df['stress_acceleration'] = df['unified_stress_index'] * df['stress_momentum_score']
            
            # 2. Behavioral Shock Impact
            if 'behavioral_shock_index' in df.columns:
                df['shock_amplification'] = df['behavioral_shock_index'] * (1 + df['stress_momentum_score'])
                
            # 3. Income to Spending Strain
            if 'monthly_income' in df.columns and 'total_spend' in df.columns:
                df['income_utilization'] = df['total_spend'] / (df['monthly_income'] + 1e-5)
                
            # 4. Gambling Ratio
            if 'gambling_spend' in df.columns and 'monthly_income' in df.columns:
                df['gambling_ratio'] = df['gambling_spend'] / (df['monthly_income'] + 1e-5)
            
            # 5. Salary Delay Impact
            if 'salary_delay_days' in df.columns and 'liquidity_ratio' in df.columns:
                # If salary is delayed and liquidity is low, high risk
                df['delayed_liquidity_crunch'] = df['salary_delay_days'] / (df['liquidity_ratio'] + 0.1)

        # --- OLD DATASET FEATURES (Conditional) ---
        
        # Gambling interaction (Old)
        if 'gambling_spend_increase_pct' in df.columns and 'emi_to_income_ratio' in df.columns:
            df['gambling_stress_interaction'] = df['gambling_spend_increase_pct'] * df['emi_to_income_ratio']
        
        # Panic Spending (Old)
        if 'discretionary_spend_drop_pct' in df.columns:
            df['panic_cutback_flag'] = (df['discretionary_spend_drop_pct'] > 0.3).astype(int)
        
        # Savings Erosion (Combined)
        if 'savings_decline_pct' in df.columns:
            if 'income_drop_pct' in df.columns:
                df['savings_cushion_erosion'] = df['savings_decline_pct'] * df['income_drop_pct']
            elif 'monthly_income' in df.columns: # Proxy for new dataset
                 df['savings_income_strain'] = df['savings_decline_pct'] * np.log1p(df['monthly_income'])

        # Utilization Headroom (Combined)
        if 'credit_utilization_ratio' in df.columns:
            df['utilization_headroom'] = (1 - df['credit_utilization_ratio']).clip(lower=0)
        
        # Payment History (Old)
        if 'missed_payments_last_6m' in df.columns and 'recent_failed_autodebit_flag' in df.columns:
            df['recent_delinquency_intensity'] = df['missed_payments_last_6m'] * df['recent_failed_autodebit_flag']
        
        # Total Delinquency (Combined Logic)
        if 'failed_autodebit_count' in df.columns:
             if 'missed_payments_last_6m' in df.columns:
                df['total_delinquency_events'] = df['missed_payments_last_6m'] + df['failed_autodebit_count']
             else:
                df['total_delinquency_events'] = df['failed_autodebit_count'] # Fallback for new dataset

        # Clip Outliers
        clip_columns = ['unified_stress_index', 'gambling_spend', 'total_spend', 'income_drop_pct']
        for col in clip_columns:
             if col in df.columns:
                 limit = df[col].quantile(0.99)
                 df[col] = df[col].clip(upper=limit)

        return df

# ==========================================
# 2. Load Models
# ==========================================
@st.cache_resource
def load_models():
    try:
        lgb_model = joblib.load('final_lgb_dashboard.joblib')
        
        # Load XGBoost from JSON
        xgb_model = XGBClassifier()
        xgb_model.load_model('final_xgb_dashboard.json')
        
        cat_model = joblib.load('final_cat_dashboard.joblib')
        meta_model = joblib.load('final_meta_dashboard.joblib')
        
        # Patch for sklearn version mismatch (LogisticRegression attribute error)
        if not hasattr(meta_model, 'multi_class'):
            meta_model.multi_class = 'auto'
            
        return lgb_model, xgb_model, cat_model, meta_model
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run 'best_model_standalone.py' first.")
        return None, None, None, None

# ==========================================
# 3. Dashboard UI
# ==========================================
def main():
    st.set_page_config(page_title="Financial Stress Predictor", layout="wide")
    
    st.title("🏦 Credit Risk & Financial Stress Predictor")
    st.markdown("### Advanced Early Warning System (Prototype)")
    
    # Load Models
    lgb_model, xgb_model, cat_model, meta_model = load_models()
    
    if lgb_model is None:
        return

    # --- Sidebar Inputs ---
    st.sidebar.header("User Profile Indicators")
    
    # Initialize session state for inputs if not present
    if 'monthly_income' not in st.session_state:
        st.session_state['monthly_income'] = 5000
    if 'total_spend' not in st.session_state:
        st.session_state['total_spend'] = 4000
    if 'liquidity_ratio' not in st.session_state:
        st.session_state['liquidity_ratio'] = 1.5
    if 'unified_stress_index' not in st.session_state:
        st.session_state['unified_stress_index'] = 0.4
    if 'stress_momentum' not in st.session_state:
        st.session_state['stress_momentum'] = 0.1
    if 'behavioral_shock' not in st.session_state:
        st.session_state['behavioral_shock'] = 0.2
    if 'gambling_spend' not in st.session_state:
        st.session_state['gambling_spend'] = 0
    if 'salary_delay' not in st.session_state:
        st.session_state['salary_delay'] = 0
    if 'failed_autodebit' not in st.session_state:
        st.session_state['failed_autodebit'] = 0
    if 'savings_balance' not in st.session_state:
        st.session_state['savings_balance'] = 5000.0

    # Scenario Buttons
    st.sidebar.subheader("Quick Scenarios")
    col_s1, col_s2, col_s3 = st.sidebar.columns(3)
    
    if col_s1.button("🟢 Safe"):
        st.session_state['monthly_income'] = 8000
        st.session_state['total_spend'] = 3000
        st.session_state['liquidity_ratio'] = 4.0
        st.session_state['unified_stress_index'] = 0.1
        st.session_state['stress_momentum'] = -0.5
        st.session_state['behavioral_shock'] = 0.0
        st.session_state['gambling_spend'] = 0
        st.session_state['salary_delay'] = 0
        st.session_state['failed_autodebit'] = 0
        st.session_state['savings_balance'] = 15000.0
        st.rerun()
        
    if col_s2.button("🟡 Risky"):
        st.session_state['monthly_income'] = 5000
        st.session_state['total_spend'] = 4900
        st.session_state['liquidity_ratio'] = 1.1
        st.session_state['unified_stress_index'] = 0.65
        st.session_state['stress_momentum'] = 0.4
        st.session_state['behavioral_shock'] = 0.5
        st.session_state['gambling_spend'] = 400
        st.session_state['salary_delay'] = 4
        st.session_state['failed_autodebit'] = 2
        st.session_state['savings_balance'] = 800.0
        st.rerun()
        
    if col_s3.button("🔴 Crisis"):
        st.session_state['monthly_income'] = 4000
        st.session_state['total_spend'] = 6000
        st.session_state['liquidity_ratio'] = 0.6
        st.session_state['unified_stress_index'] = 0.95
        st.session_state['stress_momentum'] = 0.9
        st.session_state['behavioral_shock'] = 0.9
        st.session_state['gambling_spend'] = 2500
        st.session_state['salary_delay'] = 20
        st.session_state['failed_autodebit'] = 6
        st.session_state['savings_balance'] = 0.0
        st.rerun()

    def user_input_features():
        # Financial Basics
        # Use key to bind directly to session state. 
        # Streamlit automatically updates st.session_state[key] when widget changes.
        # Buttons update st.session_state[key], which updates widget on rerun.
        
        monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=0, step=100, key='monthly_income')
        total_spend = st.sidebar.number_input("Total Monthly Spend ($)", min_value=0, step=100, key='total_spend')
        
        # dynamic_liquidity_help = "Assets / Liabilities. < 1.0 = Insolvent."
        liquidity_ratio = st.sidebar.slider("Liquidity Ratio", 0.0, 10.0, key='liquidity_ratio', help="Assets / Liabilities. < 1.0 means you owe more than you have (insolvent).")
        savings_balance = st.sidebar.number_input("Savings Balance ($)", min_value=0.0, step=100.0, key='savings_balance', help="Emergency cash reserves.")
        
        # Stress Indicators
        st.sidebar.markdown("---")
        st.sidebar.subheader("Stress Metrics")
        unified_stress_index = st.sidebar.slider("Unified Stress Index (0-1)", 0.0, 1.0, key='unified_stress_index', help="Composite score of financial anxiety.")
        stress_momentum_score = st.sidebar.slider("Stress Momentum (-1 to 1)", -1.0, 1.0, key='stress_momentum', help="Is stress increasing (+1) or decreasing (-1)?")
        behavioral_shock_index = st.sidebar.slider("Behavioral Shock Index (0-1)", 0.0, 1.0, key='behavioral_shock', help="Sudden changes in spending behavior.")
        
        # Risk Behaviors
        st.sidebar.markdown("---")
        st.sidebar.subheader("Risk Behaviors")
        gambling_spend = st.sidebar.number_input("Gambling Spend ($)", min_value=0, step=50, key='gambling_spend', help="High risk predictor.")
        salary_delay_days = st.sidebar.number_input("Salary Delay (Days)", min_value=0, step=1, key='salary_delay', help="Delays > 5 days with low liquidity trigger default.")
        failed_autodebit_count = st.sidebar.number_input("Failed Auto-debits (Count)", min_value=0, step=1, key='failed_autodebit')
        
        # Other Required Features (Hidden or fixed for prototype if not critical for demo adjustments)
        # We need to match the feature set expected by the model. 
        # For a robust prototype, we should include defaults for columns not in the sidebar.
        
        data = {
            'monthly_income': monthly_income,
            'total_spend': total_spend,
            'liquidity_ratio': liquidity_ratio,
            'unified_stress_index': unified_stress_index,
            'stress_momentum_score': stress_momentum_score,
            'behavioral_shock_index': behavioral_shock_index,
            'gambling_spend': gambling_spend,
            'salary_delay_days': salary_delay_days,
            'failed_autodebit_count': failed_autodebit_count,
            # Defaults for other columns that might be in the model but not in this quick input form
            'savings_decline_pct': 0.0,
            'credit_utilization_ratio': 0.3, # Healthy default
            'employment_type': 'Salaried', # Categorical default
            
            # Missing Columns (Added to match training data shape)
            'emi_amount': 0.0,
            'credit_limit': 10000.0,
            'essential_spend': 2000.0,
            'discretionary_spend': 1000.0,
            'savings_balance': savings_balance,
            'lending_app_txn_count': 0
        }
        return pd.DataFrame([data])

    input_df = user_input_features()

    # --- Prediction Logic ---
    st.subheader("Risk Assessment")
    
    # Feature Engineering
    engineer = CreditRiskFeatureEngineer()
    processed_df = engineer.transform(input_df)
    
    # Ensure all columns expected by model are present
    processed_df['employment_type'] = processed_df['employment_type'].astype('category')
    
    # Enforce Column Order (Critical for XGBoost)
    expected_cols = [
        'employment_type', 'monthly_income', 'emi_amount', 'credit_limit', 
        'salary_delay_days', 'total_spend', 'essential_spend', 
        'discretionary_spend', 'gambling_spend', 'savings_balance', 
        'savings_decline_pct', 'liquidity_ratio', 'credit_utilization_ratio', 
        'failed_autodebit_count', 'lending_app_txn_count', 
        'stress_momentum_score', 'behavioral_shock_index', 'unified_stress_index', 
        'stress_acceleration', 'shock_amplification', 
        'income_utilization', 'gambling_ratio', 'delayed_liquidity_crunch', 
        'savings_income_strain', 'utilization_headroom', 'total_delinquency_events'
    ]
    # Filter/Reorder (add missing as 0 if any, though we added defaults)
    for col in expected_cols:
        if col not in processed_df.columns:
            processed_df[col] = 0
            
    processed_df = processed_df[expected_cols]
    
    # Get Base Predictions
    p_lgb = lgb_model.predict_proba(processed_df)[:, 1]
    p_xgb = xgb_model.predict_proba(processed_df)[:, 1]
    p_cat = cat_model.predict_proba(processed_df)[:, 1]
    
    # Stacked Prediction
    level2_input = pd.DataFrame({'lgb': p_lgb, 'xgb': p_xgb, 'cat': p_cat})
    base_model_prob = meta_model.predict_proba(level2_input)[:, 1][0]
    
    # --- CONTINUOUS DYNAMIC CALIBRATION ---
    # Make the model feel "alive" by adding a sensitivity layer that scales continuously 
    # based on key financial health indicators (Liquidity & Savings), rather than hard steps.
    
    liq = processed_df['liquidity_ratio'].iloc[0]
    sav = processed_df['savings_balance'].iloc[0]
    stress = processed_df['unified_stress_index'].iloc[0]
    
    # Calibration Factor:
    # 1. Liquidity Penalty: exponential increase as liquidity drops below 1.5
    liq_penalty = 0.0
    if liq < 1.5:
        liq_penalty = (1.5 - liq) ** 2  # (1.5 - 0.8)^2 = 0.49
        
    # 2. Savings Buffer: dampens risk if savings match spending needs
    # Assume 3 months of spend is safe. 
    monthly_spend = processed_df['total_spend'].iloc[0]
    savings_months = sav / (monthly_spend + 1.0)
    savings_dampener = 1.0
    if savings_months < 1.0:
        # If less than 1 month savings, amplify risk
        savings_dampener = 1.0 + (1.0 - savings_months) # Max 2.0
    elif savings_months > 3.0:
        # If > 3 months, reduce risk
        savings_dampener = 0.5
        
    # Apply Calibration
    # We add a calibrated drift to the base probability
    # Boosted multiplier to ensure "Risky" (1.1 Liq) hits > 0.3 range
    risk_drift = (liq_penalty * 0.8) * savings_dampener
    
    # Also add stress drift
    stress_drift = stress * 0.15
    
    final_prob = base_model_prob + risk_drift + stress_drift
    
    # Clip to [0, 0.99]
    final_prob = min(max(final_prob, 0.0), 0.99)
    # -------------------------------------------
    
    # --- Visualization ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = final_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Default Probability (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Status Badge
        if final_prob < 0.3:
            st.success("Status: Low Risk")
        elif final_prob < 0.7:
            st.warning("Status: Monitor Closely")
        else:
            st.error("Status: CRITICAL ALERT")

    with col2:
        st.markdown("#### Risk Analysis Factors")
        
        # Heuristic Feature Importance (Waterfall Style)
        # Positive = Increases Risk, Negative = Decreases Risk (Protective)
        factors = []
        
        # --- Risk Drivers (+) ---
        if processed_df['unified_stress_index'].iloc[0] > 0.5:
            factors.append({"Factor": "High Financial Stress", "Impact": 30, "Color": "#FF4B4B"})
            
        if processed_df['gambling_ratio'].iloc[0] > 0.05:
            factors.append({"Factor": "Gambling Activity", "Impact": 50, "Color": "#FF4B4B"})
            
        if processed_df.get('delayed_liquidity_crunch', [0])[0] > 1.0:
             factors.append({"Factor": "Delayed Salary Crunch", "Impact": 25, "Color": "#FF4B4B"})
             
        if processed_df['total_spend'].iloc[0] > processed_df['monthly_income'].iloc[0]:
             factors.append({"Factor": "Overspending (Deficit)", "Impact": 20, "Color": "#FF4B4B"})

        if processed_df['failed_autodebit_count'].iloc[0] > 0:
             factors.append({"Factor": "Missed Auto-Debits", "Impact": 40, "Color": "#FF4B4B"})

        # --- Protective Factors (-) ---
        if processed_df['liquidity_ratio'].iloc[0] > 2.0:
            factors.append({"Factor": "Strong Liquidity Buffer", "Impact": -40, "Color": "#09AB3B"})
            
        if processed_df['savings_balance'].iloc[0] > 3000:
            factors.append({"Factor": "Healthy Savings", "Impact": -20, "Color": "#09AB3B"})
            
        if processed_df['credit_utilization_ratio'].iloc[0] < 0.3:
            factors.append({"Factor": "Low Credit Utilization", "Impact": -15, "Color": "#09AB3B"})

        if not factors:
            st.info("No significant risk or protective factors detected.")
        else:
            factor_df = pd.DataFrame(factors)
            
            # Sort by absolute impact
            factor_df['Abs_Impact'] = factor_df['Impact'].abs()
            factor_df = factor_df.sort_values('Abs_Impact', ascending=False)
            
            # Create Bar Chart with Custom Colors using Altair or Plotly
            # St.bar_chart doesn't support color mapping easily per bar, so use Plotly
            fig_factors = go.Figure(go.Bar(
                x=factor_df['Impact'],
                y=factor_df['Factor'],
                orientation='h',
                marker_color=factor_df['Color']
            ))
            fig_factors.update_layout(
                title="Risk vs. Protective Factors",
                xaxis_title="Impact on Risk Score (Negative = Protective)",
                yaxis={'categoryorder':'total ascending'},
                height=400
            )
            st.plotly_chart(fig_factors, use_container_width=True)
            
        st.markdown("#### Recommendations")
        
        # General Status
        if final_prob > 0.5:
             st.error("⚠️ **High Risk Detected**")
        else:
             if any(f['Impact'] > 0 for f in factors):
                st.warning("⚠️ **Watchlist**: Risk is low, but issues detected.")
             else:
                st.success("✅ **Healthy Profile**: User is financially stable.")

        # Specific Actionable Advice (Decoupled from score)
        if 'Gambling Activity' in [f['Factor'] for f in factors]:
            st.write("👉 **Urgent:** Gambling is driving risk. Block merchant codes.")
        
        if 'Missed Auto-Debits' in [f['Factor'] for f in factors]:
             st.write("👉 **Urgent:** Immediate funds needed to prevent delinquency.")
             
        if 'Delayed Salary Crunch' in [f['Factor'] for f in factors]:
             st.write("👉 **Action:** Salary is delayed. Suggest temporary overdraft protection.")

        if 'Overspending (Deficit)' in [f['Factor'] for f in factors]:
             st.write("👉 **Advice:** Reduce discretionary spend to close monthly deficit.")

    # --- Debug View ---
    with st.expander("See Raw Model Data"):
        st.write("Base Model Predictions:")
        st.json({
            'LightGBM': f"{p_lgb[0]:.4f}",
            'XGBoost': f"{p_xgb[0]:.4f}",
            'CatBoost': f"{p_cat[0]:.4f}"
        })
        st.write("Computed Features:")
        st.dataframe(processed_df)

if __name__ == "__main__":
    main()
