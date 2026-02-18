from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    mean_squared_error,
)
import numpy as np
        
def evaluate_model(model, X, y, task_type, config):
    
    if task_type == "classification":
        metric = config.primary_metric or "roc_auc"
        if metric == "roc_auc":
            scoring = "roc_auc"
            
        elif metric == "f1":
            scoring = "f1"
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        scores = cross_val_score(model, X, y, cv=config.cv_folds, scoring=scoring)
        return float(np.mean(scores))
    
    elif task_type == "regression":

        metric = config.primary_metric or "rmse"

        if metric == "rmse":
            scoring = "neg_root_mean_squared_error"
        else:
            raise ValueError(f"Unsupported regression metric: {metric}")

        scores = cross_val_score(
            model,
            X,
            y,
            cv=config.cv_folds,
            scoring=scoring
        )

        return float(np.mean(scores))
    else:
        raise ValueError(f"Unsupported task type: {task_type}")