"""
core/exceptions.py
Hierarchy
---------
SelfTrainerError               ← base for all project errors
  ├── InvalidDatasetError      ← bad input data (NaN columns, wrong shape, etc.)
  ├── InvalidConfigError       ← bad EngineConfig arguments
  ├── UnsupportedTaskError     ← task type not in {classification, regression}
  ├── UnsupportedMetricError   ← metric not supported by evaluator
  ├── ModelNotTrainedError     ← predict/explain called before fit
  ├── ExplainerNotReadyError   ← SHAP explainer used before fit
  ├── BaselineNotAvailableError← drift check before baseline built
  ├── DriftCalculationError    ← PSI calculation failed (NaN, empty data, etc.)
  └── PersistenceError         ← model save / load failure
"""


class SelfTrainerError(Exception):
    """Base class for all Self-Trainer exceptions."""


# ── Data & Config ─────────────────────────────────────────────────────────────

class InvalidDatasetError(SelfTrainerError):
    """
    Raised when the input DataFrame cannot be used for training.

    Common causes:
      - Columns contain only NaN values
      - Columns contain Inf / -Inf values
      - Target column has only one unique class
      - Dataset has fewer rows than the number of CV folds
    """


class InvalidConfigError(SelfTrainerError):
    """
    Raised when EngineConfig receives invalid arguments.

    Example: mode must be 'dev' or 'full'.
    """


# ── Task & Metric ─────────────────────────────────────────────────────────────

class UnsupportedTaskError(SelfTrainerError):
    """
    Raised when an unsupported task type is passed to the evaluator
    or model registry (anything other than 'classification' or 'regression').
    """


class UnsupportedMetricError(SelfTrainerError):
    """
    Raised when a metric string is not handled by evaluate_model.
    """


# ── Engine State ──────────────────────────────────────────────────────────────

class ModelNotTrainedError(SelfTrainerError):
    """
    Raised when predict(), summary(), or save() is called before fit().
    """


class ExplainerNotReadyError(SelfTrainerError):
    """
    Raised when explain_global() or explain_instance() is called
    before the SHAP explainer has been initialised.
    """


class BaselineNotAvailableError(SelfTrainerError):
    """
    Raised when check_drift() is called before fit() has built
    the training-data baseline.
    """


# ── Monitoring ────────────────────────────────────────────────────────────────

class DriftCalculationError(SelfTrainerError):
    """
    Raised when PSI cannot be computed.

    Common causes:
      - New data is entirely outside the baseline bin edges
        (all counts = 0, division by zero → NaN PSI)
      - New DataFrame is empty
      - New DataFrame contains only NaN values
    """


# ── Persistence ───────────────────────────────────────────────────────────────

class PersistenceError(SelfTrainerError):
    """
    Raised when model serialisation or deserialisation fails.
    """