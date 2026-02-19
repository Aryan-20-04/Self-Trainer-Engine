"""
core/config.py
==============
Central configuration object for SelfTrainerEngine.
"""

import logging
from core.exceptions import InvalidConfigError

logger = logging.getLogger("self_trainer.core.config")


class EngineConfig:

    def __init__(self, mode: str = "dev"):

        if mode == "dev":
            self.cv_folds = 3
            self.optuna_timeout = 20
            self.shap_sample_size = 500
            self.tree_estimators = 100

        elif mode == "full":
            self.cv_folds = 5
            self.optuna_timeout = 120
            self.shap_sample_size = 1000
            self.tree_estimators = 300

        else:
            raise InvalidConfigError(
                f"Mode must be 'dev' or 'full', got: {mode!r}"
            )

        self.mode = mode

        # ── Drift ─────────────────────────────────────────────────────────
        self.drift_threshold = 0.2

        # ── Primary metric (auto-detected per task if None) ────────────────
        self.primary_metric = None
        self.classification_metric = "roc_auc"
        self.regression_metric = "neg_root_mean_squared_error"

        logger.info("EngineConfig initialised (mode=%s).", mode)