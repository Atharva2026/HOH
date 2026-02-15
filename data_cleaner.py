
import pandas as pd
from data_processor import get_processed_data

def clean_data(df, target_col="default_next_30_days"):
    """
    Filters out contradictory samples from the dataset.
    Specifically removes cases where users have perfect records but are labeled as defaulters.
    """
    initial_count = len(df)
    
    # Define "Perfect Record" criteria
    # No missed payments, no failed debits, and good credit score
    contradiction_mask = (
        (df[target_col] == 1) & 
        (df['missed_payments_last_6m'] == 0) & 
        (df['failed_autodebit_count'] == 0) & 
        (df['credit_score'] > 700)
    )
    
    df_cleaned = df[~contradiction_mask].copy()
    removed_count = contradiction_mask.sum()
    
    print(f"\n--- Data Cleaning Report ---")
    print(f"Initial Rows: {initial_count}")
    print(f"Contradictory Samples Removed: {removed_count} ({removed_count/initial_count:.2%})")
    print(f"Final Rows: {len(df_cleaned)}")
    
    return df_cleaned

def get_cleaned_processed_data(filepath="data.csv"):
    """
    Pipeline: Load -> Clean -> Process Features
    """
    # 1. Load Raw Data
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None, None

    target_col = "default_next_30_days"
    
    # 2. Clean Data (Remove impossible samples)
    df_clean = clean_data(df, target_col)
    
    # 3. Process Features (using logic from data_processor.py)
    # We need to adapt data_processor to accept a dataframe, not just efficient loading
    # Ideally data_processor.get_processed_data handles file loading. 
    # Let's instantiate the transformer directly.
    
    from data_processor import CreditRiskFeatureEngineer
    
    y = df_clean[target_col]
    X_raw = df_clean.drop(target_col, axis=1)
    
    engineer = CreditRiskFeatureEngineer()
    X_processed = engineer.transform(X_raw)
    
    return X_processed, y
