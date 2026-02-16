import pandas as pd

def detect_task_type(model):
    if pd.api.types.is_numeric_dtype(model):
        unique_values = model.nunique()
        
        if unique_values <=20  and all(float(val).is_integer() for val in model.unique()):
            return 'classification'
        else:
            return 'regression'
    else:
        return 'classification'