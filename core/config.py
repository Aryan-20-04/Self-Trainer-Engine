class EngineConfig:

    def __init__(self, mode="dev"):

        self.mode = mode

        # =========================
        # Dev vs Full Mode
        # =========================
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
            raise ValueError("Mode must be 'dev' or 'full'")

        # =========================
        # Drift Settings
        # =========================
        self.drift_threshold = 0.2

        # =========================
        # Primary Metric (Used Everywhere)
        # =========================
        self.primary_metric = None  # Auto-detect based on task

        # Fallback metrics
        self.classification_metric = "roc_auc"
        self.regression_metric = "neg_root_mean_squared_error"
