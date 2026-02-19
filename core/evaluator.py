"""
core/evaluator.py
=================
Cross-validates a model and returns its mean score.

Changes (Steps 2 & 3):
  - ValueError replaced with UnsupportedTaskError / UnsupportedMetricError.
  - Logging added at DEBUG level for each CV score.
"""

import logging
import numpy as np
from sklearn.model_selection import cross_val_score

from core.exceptions import UnsupportedTaskError, UnsupportedMetricError

logger = logging.getLogger("self_trainer.core.evaluator")


def evaluate_model(model, X, y, task_type: str, config) -> float:
    """
    Run stratified cross-validation and return the mean score.

    Parameters
    ----------
    model : sklearn estimator
    X : array-like
    y : array-like
    task_type : str
        'classification' or 'regression'
    config : EngineConfig

    Returns
    -------
    float
        Mean CV score across folds.

    Raises
    ------
    UnsupportedTaskError
        If task_type is not 'classification' or 'regression'.
    UnsupportedMetricError
        If the metric string in config is not handled.
    """
    if task_type == "classification":
        metric = config.primary_metric or "roc_auc"

        if metric == "roc_auc":
            scoring = "roc_auc"
        elif metric == "f1":
            scoring = "f1"
        else:
            raise UnsupportedMetricError(
                f"Unsupported classification metric: '{metric}'. "
                f"Supported: 'roc_auc', 'f1'."
            )

        scores = cross_val_score(model, X, y, cv=config.cv_folds, scoring=scoring)
        mean_score = float(np.mean(scores))
        logger.debug("Classification CV (%s): %.4f ± %.4f", scoring, mean_score, np.std(scores))
        return mean_score

    elif task_type == "regression":
        metric = config.primary_metric or "rmse"

        if metric == "rmse":
            scoring = "neg_root_mean_squared_error"
        else:
            raise UnsupportedMetricError(
                f"Unsupported regression metric: '{metric}'. "
                f"Supported: 'rmse'."
            )

        scores = cross_val_score(model, X, y, cv=config.cv_folds, scoring=scoring)
        mean_score = float(np.mean(scores))
        logger.debug("Regression CV (%s): %.4f ± %.4f", scoring, mean_score, np.std(scores))
        return mean_score

    else:
        raise UnsupportedTaskError(
            f"Unsupported task type: '{task_type}'. "
            f"Supported: 'classification', 'regression'."
        )