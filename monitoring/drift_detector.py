import numpy as np
import pandas as pd

def calculate_psi(expected, actual, bins=10):
    expected = np.array(expected)
    actual = np.array(actual)
    
    excepted_counts, bin_edges = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual,bins=bin_edges)
    
    expected_perc = excepted_counts / len(expected)
    actual_perc = actual_counts / len(actual)
    
    psi = np.sum(
        (actual_perc - expected_perc) *
        np.log((actual_perc+1e-8) / (expected_perc+1e-8))
    )
    
    return psi

def detect_drift(reference_df, new_df, threshold=0.2):
    
    drift_report = {}
    
    for col in reference_df.columns:
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            psi = calculate_psi(reference_df[col], new_df[col])
            drift_report[col] = psi
            
    drifted_features = {
        k: v for k,v in drift_report.items() if v > threshold
    }
    
    return drift_report, drifted_features