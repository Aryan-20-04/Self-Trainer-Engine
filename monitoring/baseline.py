import numpy as np
import pandas as pd

def build_baseline(reference_df, bins=10):
    
    baseline = {}
    for col in reference_df.columns:
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            counts, bin_edges = np.histogram(
                reference_df[col],
                bins=bins
            )
            
            percentages = counts / counts.sum()
            
            baseline[col] = {
                "bin_edges": bin_edges.tolist(),
                "percentages": percentages.tolist()   
            }
    return baseline