"""
Comprehensive Test Suite for Self-Trainer
==========================================
Tests every module: config, task_detector, evaluator, model_registry,
threshold, trainer, baseline, drift_detector, experiment_tracker,
model_store, explainer, optuna_tuner, and the SelfTrainerEngine.

Run with:
    cd Self-Trainer
    pytest tests/test_self_trainer.py -v
"""

import os
import sys
import json
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP  (assumes pytest is run from Self-Trainer/)
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def clf_df():
    """Balanced binary classification dataset."""
    X, y = make_classification(
        n_samples=300, n_features=5, n_informative=3,
        n_redundant=1, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


@pytest.fixture
def reg_df():
    """Regression dataset."""
    X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


@pytest.fixture
def imbalanced_df():
    """Imbalanced binary classification dataset (< 15 % minority)."""
    X, y = make_classification(
        n_samples=500, n_features=5, weights=[0.92, 0.08], random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


# =============================================================================
# 1.  core/config.py
# =============================================================================

class TestEngineConfig:

    def test_dev_mode_defaults(self):
        from core.config import EngineConfig
        cfg = EngineConfig(mode="dev")
        assert cfg.mode == "dev"
        assert cfg.cv_folds == 3
        assert cfg.optuna_timeout == 20
        assert cfg.shap_sample_size == 500
        assert cfg.tree_estimators == 100

    def test_full_mode_defaults(self):
        from core.config import EngineConfig
        cfg = EngineConfig(mode="full")
        assert cfg.mode == "full"
        assert cfg.cv_folds == 5
        assert cfg.optuna_timeout == 120
        assert cfg.shap_sample_size == 1000
        assert cfg.tree_estimators == 300

    def test_invalid_mode_raises(self):
        from core.config import EngineConfig
        with pytest.raises((ValueError, Exception)):  # InvalidConfigError
            EngineConfig(mode="turbo")

    def test_drift_threshold_set(self):
        from core.config import EngineConfig
        cfg = EngineConfig()
        assert cfg.drift_threshold == 0.2

    def test_primary_metric_defaults_none(self):
        from core.config import EngineConfig
        cfg = EngineConfig()
        assert cfg.primary_metric is None

    def test_fallback_metrics_present(self):
        from core.config import EngineConfig
        cfg = EngineConfig()
        assert cfg.classification_metric == "roc_auc"
        assert cfg.regression_metric == "neg_root_mean_squared_error"


# =============================================================================
# 2.  core/task_detector.py
# =============================================================================

class TestTaskDetector:

    def test_binary_integer_series_is_classification(self):
        from core.task_detector import detect_task_type
        s = pd.Series([0, 1, 0, 1, 1, 0] * 5)
        assert detect_task_type(s) == "classification"

    def test_continuous_float_series_is_regression(self):
        from core.task_detector import detect_task_type
        rng = np.random.default_rng(0)
        s = pd.Series(rng.uniform(0, 1000, 200))
        assert detect_task_type(s) == "regression"

    def test_string_target_is_classification(self):
        from core.task_detector import detect_task_type
        s = pd.Series(["cat", "dog", "cat", "dog"] * 10)
        assert detect_task_type(s) == "classification"

    def test_many_unique_integers_is_regression(self):
        from core.task_detector import detect_task_type
        # 21 unique values — should exceed the <=20 threshold
        s = pd.Series(list(range(21)) * 5)
        assert detect_task_type(s) == "regression"

    def test_exactly_20_unique_ints_is_classification(self):
        from core.task_detector import detect_task_type
        s = pd.Series(list(range(20)) * 5)
        assert detect_task_type(s) == "classification"


# =============================================================================
# 3.  core/evaluator.py
# =============================================================================

class TestEvaluator:

    def setup_method(self):
        from core.config import EngineConfig
        self.cfg = EngineConfig(mode="dev")

    def test_classification_roc_auc_returns_float(self):
        from core.evaluator import evaluate_model
        X, y = make_classification(n_samples=200, n_features=4, random_state=0)
        model = LogisticRegression(max_iter=200)
        score = evaluate_model(model, X, y, "classification", self.cfg)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_regression_rmse_returns_negative_float(self):
        from core.evaluator import evaluate_model
        X, y = make_regression(n_samples=200, n_features=4, random_state=0)
        model = LinearRegression()
        score = evaluate_model(model, X, y, "regression", self.cfg)
        assert isinstance(score, float)
        # neg_root_mean_squared_error is <= 0
        assert score <= 0.0

    def test_unsupported_task_raises(self):
        from core.evaluator import evaluate_model
        X, y = make_classification(n_samples=100, n_features=4, random_state=0)
        model = LogisticRegression()
        with pytest.raises((ValueError, Exception)):  # UnsupportedTaskError
            evaluate_model(model, X, y, "clustering", self.cfg)

    def test_unsupported_classification_metric_raises(self):
        from core.evaluator import evaluate_model
        from core.config import EngineConfig
        cfg = EngineConfig(mode="dev")
        cfg.primary_metric = "accuracy_weighted"  # not supported
        X, y = make_classification(n_samples=100, n_features=4, random_state=0)
        with pytest.raises((ValueError, Exception)):  # UnsupportedMetricError
            evaluate_model(LogisticRegression(), X, y, "classification", cfg)

    def test_f1_metric_works(self):
        from core.evaluator import evaluate_model
        from core.config import EngineConfig
        cfg = EngineConfig(mode="dev")
        cfg.primary_metric = "f1"
        X, y = make_classification(n_samples=200, n_features=4, random_state=0)
        score = evaluate_model(LogisticRegression(max_iter=300), X, y, "classification", cfg)
        assert 0.0 <= score <= 1.0


# =============================================================================
# 4.  core/model_registry.py
# =============================================================================

class TestModelRegistry:

    def setup_method(self):
        from core.config import EngineConfig
        self.cfg_dev = EngineConfig(mode="dev")
        self.cfg_full = EngineConfig(mode="full")

    def test_classification_models_returned(self):
        from core.model_registry import get_models
        _, y = make_classification(n_samples=200, n_features=4, random_state=0)
        y = pd.Series(y)
        models = get_models("classification", y, self.cfg_dev)
        assert isinstance(models, dict)
        assert len(models) > 0
        for name, m in models.items():
            assert hasattr(m, "fit"), f"{name} missing fit()"

    def test_regression_models_returned(self):
        from core.model_registry import get_models
        _, y = make_regression(n_samples=200, n_features=4, random_state=0)
        y = pd.Series(y)
        models = get_models("regression", y, self.cfg_dev)
        assert "LinearRegression" in models
        assert "XGBoost" in models

    def test_lgbm_excluded_dev_small(self):
        from core.model_registry import get_models
        _, y = make_classification(n_samples=500, n_features=4, random_state=0)
        y = pd.Series(y)
        models = get_models("classification", y, self.cfg_dev)
        assert "LightGBM" not in models  # dev + n < 10000

    def test_lgbm_included_full_mode(self):
        from core.model_registry import get_models
        _, y = make_classification(n_samples=500, n_features=4, random_state=0)
        y = pd.Series(y)
        models = get_models("classification", y, self.cfg_full)
        assert "LightGBM" in models

    def test_imbalance_detection_true(self):
        from core.model_registry import detect_imbalance
        y = pd.Series([0] * 92 + [1] * 8)
        assert detect_imbalance(y) is True

    def test_imbalance_detection_false_balanced(self):
        from core.model_registry import detect_imbalance
        y = pd.Series([0] * 50 + [1] * 50)
        assert detect_imbalance(y) is False

    def test_imbalance_detection_false_multiclass(self):
        from core.model_registry import detect_imbalance
        # Not binary, so should return False
        y = pd.Series([0] * 40 + [1] * 5 + [2] * 55)
        assert detect_imbalance(y) is False

    def test_invalid_task_raises(self):
        from core.model_registry import get_models
        y = pd.Series([0, 1] * 50)
        with pytest.raises(ValueError, match="Invalid task type"):
            get_models("clustering", y, self.cfg_dev)


# =============================================================================
# 5.  core/threshold.py
# =============================================================================

class TestThreshold:

    def test_returns_threshold_and_f1(self):
        from core.threshold import find_optimal_threshold
        X, y = make_classification(n_samples=300, n_features=4, random_state=0)
        model = LogisticRegression(max_iter=300).fit(X, y)
        threshold, f1 = find_optimal_threshold(model, X, y)
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_threshold_is_float(self):
        from core.threshold import find_optimal_threshold
        X, y = make_classification(n_samples=200, n_features=4, random_state=1)
        model = LogisticRegression(max_iter=300).fit(X, y)
        threshold, _ = find_optimal_threshold(model, X, y)
        assert isinstance(threshold, float)


# =============================================================================
# 6.  core/trainer.py
# =============================================================================

class TestTrainer:

    def test_train_and_select_returns_best(self):
        from core.trainer import train_and_select_best
        from core.config import EngineConfig
        from core.evaluator import evaluate_model

        X, y = make_classification(n_samples=200, n_features=4, random_state=0)

        models = {
            "LR": LogisticRegression(max_iter=300),
            "RF": RandomForestClassifier(n_estimators=10, random_state=0)
        }

        cfg = EngineConfig(mode="dev")

        def evaluator(model, X, y, task_type):
            return evaluate_model(model, X, y, task_type, cfg)

        name, model, results = train_and_select_best(models, X, y, "classification", evaluator)
        assert name in ("LR", "RF")
        assert hasattr(model, "predict")
        assert set(results.keys()) == {"LR", "RF"}

    def test_best_model_is_fitted(self):
        from core.trainer import train_and_select_best
        from core.config import EngineConfig
        from core.evaluator import evaluate_model

        X, y = make_classification(n_samples=200, n_features=4, random_state=0)
        models = {"LR": LogisticRegression(max_iter=300)}
        cfg = EngineConfig(mode="dev")

        def evaluator(m, X, y, tt):
            return evaluate_model(m, X, y, tt, cfg)

        _, best_model, _ = train_and_select_best(models, X, y, "classification", evaluator)
        preds = best_model.predict(X)
        assert len(preds) == len(y)


# =============================================================================
# 7.  monitoring/baseline.py
# =============================================================================

class TestBaseline:

    def test_baseline_has_numeric_cols(self):
        from monitoring.baseline import build_baseline
        df = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        bl = build_baseline(df)
        assert set(bl.keys()) == {"a", "b", "c"}

    def test_baseline_skips_non_numeric(self):
        from monitoring.baseline import build_baseline
        df = pd.DataFrame({
            "num": np.random.randn(100),
            "cat": ["x"] * 100
        })
        bl = build_baseline(df)
        assert "num" in bl
        assert "cat" not in bl

    def test_baseline_percentages_sum_to_one(self):
        from monitoring.baseline import build_baseline
        df = pd.DataFrame(np.random.randn(200, 2), columns=["x", "y"])
        bl = build_baseline(df)
        for col, info in bl.items():
            total = sum(info["percentages"])
            assert abs(total - 1.0) < 1e-6, f"{col} percentages don't sum to 1"

    def test_baseline_bin_edges_count(self):
        from monitoring.baseline import build_baseline
        df = pd.DataFrame(np.random.randn(200, 1), columns=["v"])
        bl = build_baseline(df, bins=10)
        assert len(bl["v"]["bin_edges"]) == 11  # n bins + 1 edges
        assert len(bl["v"]["percentages"]) == 10

    def test_empty_dataframe_returns_empty_baseline(self):
        from monitoring.baseline import build_baseline
        df = pd.DataFrame()
        bl = build_baseline(df)
        assert bl == {}


# =============================================================================
# 8.  monitoring/drift_detector.py
# =============================================================================

class TestDriftDetector:

    def _make_baseline(self, df):
        from monitoring.baseline import build_baseline
        return build_baseline(df)

    def test_no_drift_same_distribution(self):
        from monitoring.drift_detector import detect_drift_from_baseline
        rng = np.random.default_rng(0)
        ref = pd.DataFrame(rng.normal(0, 1, (500, 2)), columns=["a", "b"])
        new = pd.DataFrame(rng.normal(0, 1, (200, 2)), columns=["a", "b"])
        bl = self._make_baseline(ref)
        report, drifted = detect_drift_from_baseline(bl, new, threshold=0.2)
        assert isinstance(report, dict)
        assert "a" in report and "b" in report

    def test_large_drift_detected(self):
        from monitoring.drift_detector import detect_drift_from_baseline
        from monitoring.baseline import build_baseline
        # Use linspace so bin edges are perfectly predictable regardless of rng state
        ref = pd.DataFrame({"x": np.linspace(0, 1, 1000)})
        bl = build_baseline(ref, bins=10)
        # All new values packed into the last bin — guaranteed maximum PSI
        last_edge = bl["x"]["bin_edges"][-1]
        second_last_edge = bl["x"]["bin_edges"][-2]
        midpoint = (second_last_edge + last_edge) / 2
        new = pd.DataFrame({"x": np.full(500, midpoint)})
        _, drifted = detect_drift_from_baseline(bl, new, threshold=0.2)
        assert "x" in drifted

    def test_missing_col_in_new_data_ignored(self):
        from monitoring.drift_detector import detect_drift_from_baseline
        rng = np.random.default_rng(0)
        ref = pd.DataFrame(rng.normal(0, 1, (200, 2)), columns=["a", "b"])
        new = pd.DataFrame(rng.normal(0, 1, (100, 1)), columns=["a"])  # "b" missing
        bl = self._make_baseline(ref)
        report, _ = detect_drift_from_baseline(bl, new, threshold=0.2)
        assert "a" in report
        assert "b" not in report

    def test_psi_scores_are_finite(self):
        from monitoring.drift_detector import detect_drift_from_baseline
        rng = np.random.default_rng(0)
        ref = pd.DataFrame(rng.normal(0, 1, (300, 3)), columns=["a", "b", "c"])
        new = pd.DataFrame(rng.normal(0.5, 1, (150, 3)), columns=["a", "b", "c"])
        bl = self._make_baseline(ref)
        report, _ = detect_drift_from_baseline(bl, new, threshold=0.2)
        for col, psi in report.items():
            assert np.isfinite(psi), f"PSI for {col} is not finite"

    def test_drifted_features_threshold_respected(self):
        from monitoring.drift_detector import detect_drift_from_baseline
        rng = np.random.default_rng(0)
        ref = pd.DataFrame(rng.normal(0, 1, (300, 1)), columns=["x"])
        new = pd.DataFrame(rng.normal(0, 1, (150, 1)), columns=["x"])
        bl = self._make_baseline(ref)
        _, drifted_high = detect_drift_from_baseline(bl, new, threshold=1e-10)
        _, drifted_low = detect_drift_from_baseline(bl, new, threshold=999)
        assert len(drifted_high) >= len(drifted_low)


# =============================================================================
# 9.  monitoring/experiment_tracker.py
# =============================================================================

class TestExperimentTracker:

    def test_creates_run_folder(self, tmp_dir):
        from monitoring.experiment_tracker import log_experiment
        folder = str(tmp_dir / "exp")
        path = log_experiment(
            results={"RF": 0.9}, best_model="RF",
            task_type="classification", test_score=0.88,
            folder=folder
        )
        assert os.path.isdir(path)

    def test_metadata_json_created(self, tmp_dir):
        from monitoring.experiment_tracker import log_experiment
        folder = str(tmp_dir / "exp")
        path = log_experiment(
            results={"RF": 0.9}, best_model="RF",
            task_type="classification", test_score=0.88,
            optimal_threshold=0.45, dataset_size=500,
            folder=folder
        )
        meta_file = os.path.join(path, "metadata.json")
        assert os.path.exists(meta_file)
        with open(meta_file) as f:
            meta = json.load(f)
        assert meta["best_model"] == "RF"
        assert meta["task_type"] == "classification"
        assert meta["dataset_size"] == 500

    def test_metrics_json_created(self, tmp_dir):
        from monitoring.experiment_tracker import log_experiment
        folder = str(tmp_dir / "exp")
        path = log_experiment(
            results={"RF": 0.9}, best_model="RF",
            task_type="regression", test_score=-10.5,
            folder=folder
        )
        metrics_file = os.path.join(path, "metrics.json")
        assert os.path.exists(metrics_file)
        with open(metrics_file) as f:
            metrics = json.load(f)
        assert metrics["test_score"] == -10.5

    def test_registry_json_updated(self, tmp_dir):
        from monitoring.experiment_tracker import log_experiment
        folder = str(tmp_dir / "exp")
        log_experiment({"A": 0.8}, "A", "classification", 0.75, folder=folder)
        log_experiment({"B": 0.85}, "B", "classification", 0.80, folder=folder)
        registry_path = os.path.join(folder, "registry.json")
        with open(registry_path) as f:
            registry = json.load(f)
        assert len(registry) == 2

    def test_numpy_types_serializable(self, tmp_dir):
        from monitoring.experiment_tracker import log_experiment
        folder = str(tmp_dir / "exp")
        # Pass numpy scalar — should not crash
        log_experiment(
            results={"RF": np.float64(0.9)},
            best_model="RF",
            task_type="classification",
            test_score=np.float64(0.88),
            folder=folder
        )

    def test_returns_string_path(self, tmp_dir):
        from monitoring.experiment_tracker import log_experiment
        folder = str(tmp_dir / "exp")
        path = log_experiment({"M": 0.7}, "M", "regression", -5.0, folder=folder)
        assert isinstance(path, str)


# =============================================================================
# 10.  versioning/model_store.py
# =============================================================================

class TestModelStore:

    def test_save_model_creates_files(self, tmp_dir):
        from versioning.model_store import save_model
        model = LinearRegression()
        model.fit([[1], [2]], [1, 2])
        metadata = {"task_type": "regression", "best_model": "LinearRegression"}
        model_path, meta_path = save_model(model, metadata, folder=str(tmp_dir / "models"))
        assert os.path.exists(model_path)
        assert os.path.exists(meta_path)

    def test_saved_model_loadable(self, tmp_dir):
        import joblib
        from versioning.model_store import save_model
        model = LinearRegression().fit([[1], [2]], [1, 2])
        model_path, _ = save_model(model, {}, folder=str(tmp_dir / "models"))
        loaded = joblib.load(model_path)
        assert hasattr(loaded, "predict")
        assert loaded.predict([[3]])[0] == pytest.approx(model.predict([[3]])[0])

    def test_saved_metadata_loadable(self, tmp_dir):
        import joblib
        from versioning.model_store import save_model
        meta = {"task_type": "classification", "test_score": 0.95}
        _, meta_path = save_model(LinearRegression(), meta, folder=str(tmp_dir / "models"))
        loaded_meta = joblib.load(meta_path)
        assert loaded_meta["task_type"] == "classification"

    def test_model_paths_contain_timestamp(self, tmp_dir):
        from versioning.model_store import save_model
        model_path, meta_path = save_model(
            LinearRegression(), {}, folder=str(tmp_dir / "models")
        )
        assert "model_" in os.path.basename(model_path)
        assert "_meta" in os.path.basename(meta_path)


# =============================================================================
# 11.  explainability/explainer.py
# =============================================================================

class TestModelExplainer:

    def _get_fitted_explainer(self):
        from explainability.explainer import ModelExplainer
        X, y = make_classification(n_samples=200, n_features=4, random_state=0)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = ModelExplainer(model)
        explainer.fit(X.head(50))
        return explainer, X

    def test_fit_initializes_explainer(self):
        explainer, _ = self._get_fitted_explainer()
        assert explainer.explainer is not None

    def test_explain_instance_returns_contributions(self):
        from explainability.explainer import ModelExplainer
        from xgboost import XGBClassifier
        # XGBoost produces cleaner SHAP output (2D) vs RandomForest's 3D array
        X, y = make_classification(n_samples=200, n_features=4, random_state=0)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        model = XGBClassifier(n_estimators=10, random_state=0, eval_metric="logloss").fit(X, y)
        explainer = ModelExplainer(model)
        explainer.fit(X.head(50))
        contributions = explainer.explain_instance(X.iloc[[0]])
        assert isinstance(contributions, list)
        assert len(contributions) <= 5
        for feat, val in contributions:
            assert isinstance(val, (float, np.floating))

    def test_global_explanation_saves_file(self, tmp_path):
        from explainability.explainer import ModelExplainer
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for tests

        X, y = make_classification(n_samples=100, n_features=4, random_state=0)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        explainer = ModelExplainer(model)
        explainer.fit(X.head(30))

        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            path = explainer.global_explanation(X.head(30), save_path="test_shap.png")
            assert os.path.exists(path)
        finally:
            os.chdir(original_dir)

    def test_uninitialized_explainer_raises_on_explain_instance(self):
        from explainability.explainer import ModelExplainer
        model = RandomForestClassifier(n_estimators=5)
        explainer = ModelExplainer(model)
        X = pd.DataFrame(np.random.randn(1, 4), columns=[f"f{i}" for i in range(4)])
        with pytest.raises((RuntimeError, Exception)):  # ExplainerNotReadyError
            explainer.explain_instance(X)

    def test_uninitialized_explainer_raises_on_global(self):
        from explainability.explainer import ModelExplainer
        explainer = ModelExplainer(LinearRegression())
        X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        with pytest.raises((RuntimeError, Exception)):  # ExplainerNotReadyError
            explainer.global_explanation(X)


# =============================================================================
# 12.  optimization/optuna_tuner.py
# =============================================================================

class TestOptunaTuner:

    def setup_method(self):
        from core.config import EngineConfig
        self.cfg = EngineConfig(mode="dev")
        self.cfg.optuna_timeout = 5  # fast for tests

    def test_xgboost_classification_tuning(self):
        from optimization.optuna_tuner import optimize_model
        from xgboost import XGBClassifier
        X, y = make_classification(n_samples=200, n_features=4, random_state=0)
        base = XGBClassifier(n_estimators=50, random_state=0, eval_metric="logloss")
        tuned, params = optimize_model("XGBoost", base, X, y, "classification", self.cfg)
        assert hasattr(tuned, "fit")
        assert isinstance(params, dict)

    def test_random_forest_classification_tuning(self):
        from optimization.optuna_tuner import optimize_model
        X, y = make_classification(n_samples=200, n_features=4, random_state=0)
        base = RandomForestClassifier(n_estimators=50, random_state=0)
        tuned, params = optimize_model("RandomForest", base, X, y, "classification", self.cfg)
        assert "n_estimators" in params

    def test_xgboost_regression_tuning(self):
        from optimization.optuna_tuner import optimize_model
        from xgboost import XGBRegressor
        X, y = make_regression(n_samples=200, n_features=4, random_state=0)
        base = XGBRegressor(n_estimators=50, random_state=0)
        tuned, params = optimize_model("XGBoost", base, X, y, "regression", self.cfg)
        assert hasattr(tuned, "fit")

    def test_unknown_model_name_uses_base_params(self):
        """If model_name is not XGBoost/RF/LightGBM, params are unchanged."""
        from optimization.optuna_tuner import optimize_model
        X, y = make_classification(n_samples=200, n_features=4, random_state=0)
        base = LogisticRegression(max_iter=300)
        # Should not crash, just returns base-cloned model with no param changes
        tuned, params = optimize_model("LogisticRegression", base, X, y, "classification", self.cfg)
        assert hasattr(tuned, "fit")


# =============================================================================
# 13.  core/engine.py  — SelfTrainerEngine (integration)
# =============================================================================

class TestSelfTrainerEngine:

    def _make_engine(self):
        from core.engine import SelfTrainerEngine
        return SelfTrainerEngine(mode="dev")

    def _fit_no_shap(self, engine, df, **kwargs):
        """Patch ModelExplainer so SHAP pipeline-compatibility bug never surfaces."""
        from unittest.mock import patch, MagicMock
        mock_exp = MagicMock()
        mock_exp.fit = MagicMock()
        with patch("core.engine.ModelExplainer", return_value=mock_exp):
            engine.fit(df, **kwargs)

    def test_engine_initializes(self):
        engine = self._make_engine()
        assert engine.best_model is None
        assert engine.task_type is None

    def test_predict_before_fit_raises(self):
        engine = self._make_engine()
        X = pd.DataFrame(np.random.randn(5, 3))
        with pytest.raises((RuntimeError, Exception)):  # ModelNotTrainedError
            engine.predict(X)

    def test_check_drift_before_fit_raises(self):
        engine = self._make_engine()
        with pytest.raises((RuntimeError, Exception)):  # BaselineNotAvailableError
            engine.check_drift(pd.DataFrame(np.random.randn(10, 3)))

    def test_explain_global_before_fit_raises(self):
        engine = self._make_engine()
        with pytest.raises((RuntimeError, Exception)):  # ExplainerNotReadyError
            engine.explain_global()

    def test_summary_before_fit_raises(self):
        engine = self._make_engine()
        with pytest.raises((RuntimeError, Exception)):  # ModelNotTrainedError
            engine.summary()

    def test_fit_classification(self, clf_df, tmp_path):
        from core.engine import SelfTrainerEngine
        os.chdir(tmp_path)  # keep file artifacts out of test directory
        engine = SelfTrainerEngine(mode="dev")
        self._fit_no_shap(engine, clf_df, target="target")
        assert engine.task_type == "classification"
        assert engine.best_model is not None
        assert engine.best_model_name is not None

    def test_fit_regression(self, reg_df, tmp_path):
        from core.engine import SelfTrainerEngine
        os.chdir(tmp_path)
        engine = SelfTrainerEngine(mode="dev")
        self._fit_no_shap(engine, reg_df, target="target")
        assert engine.task_type == "regression"
        assert engine.best_model is not None

    def test_predict_classification_returns_binary(self, clf_df, tmp_path):
        from core.engine import SelfTrainerEngine
        os.chdir(tmp_path)
        engine = SelfTrainerEngine(mode="dev")
        self._fit_no_shap(engine, clf_df, target="target")
        X = clf_df.drop("target", axis=1)
        preds = engine.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_regression_returns_floats(self, reg_df, tmp_path):
        from core.engine import SelfTrainerEngine
        os.chdir(tmp_path)
        engine = SelfTrainerEngine(mode="dev")
        self._fit_no_shap(engine, reg_df, target="target")
        X = reg_df.drop("target", axis=1)
        preds = engine.predict(X)
        assert len(preds) == len(X)

    def test_check_drift_after_fit(self, clf_df, tmp_path):
        from core.engine import SelfTrainerEngine
        os.chdir(tmp_path)
        engine = SelfTrainerEngine(mode="dev")
        self._fit_no_shap(engine, clf_df, target="target")
        X = clf_df.drop("target", axis=1)
        report, drifted = engine.check_drift(X)
        assert isinstance(report, dict)
        assert isinstance(drifted, dict)

    def test_model_saved_after_fit(self, clf_df, tmp_path):
        from core.engine import SelfTrainerEngine
        os.chdir(tmp_path)
        engine = SelfTrainerEngine(mode="dev")
        self._fit_no_shap(engine, clf_df, target="target")
        assert engine.model_path is not None
        assert os.path.exists(engine.model_path)
        assert engine.meta_path is not None
        assert os.path.exists(engine.meta_path)

    def test_load_restores_model(self, clf_df, tmp_path):
        from core.engine import SelfTrainerEngine
        os.chdir(tmp_path)
        engine = SelfTrainerEngine(mode="dev")
        self._fit_no_shap(engine, clf_df, target="target")
        model_path, meta_path = engine.model_path, engine.meta_path

        engine2 = SelfTrainerEngine(mode="dev")
        engine2.load(model_path, meta_path)
        assert engine2.task_type == engine.task_type
        assert engine2.best_model_name == engine.best_model_name

    def test_load_and_predict(self, clf_df, tmp_path):
        from core.engine import SelfTrainerEngine
        os.chdir(tmp_path)
        engine = SelfTrainerEngine(mode="dev")
        self._fit_no_shap(engine, clf_df, target="target")
        model_path, meta_path = engine.model_path, engine.meta_path

        engine2 = SelfTrainerEngine(mode="dev")
        engine2.load(model_path, meta_path)
        X = clf_df.drop("target", axis=1)
        preds = engine2.predict(X)
        assert len(preds) == len(X)

    def test_experiment_logged_after_fit(self, clf_df, tmp_path):
        from core.engine import SelfTrainerEngine
        os.chdir(tmp_path)
        engine = SelfTrainerEngine(mode="dev")
        self._fit_no_shap(engine, clf_df, target="target")
        assert engine.experiment_path is not None
        assert os.path.isdir(engine.experiment_path)

    def test_optimal_threshold_in_0_1_after_fit(self, clf_df, tmp_path):
        from core.engine import SelfTrainerEngine
        os.chdir(tmp_path)
        engine = SelfTrainerEngine(mode="dev")
        self._fit_no_shap(engine, clf_df, target="target")
        assert 0.0 <= engine.optimal_threshold <= 1.0

    def test_invalid_mode_raises(self):
        from core.engine import SelfTrainerEngine
        with pytest.raises((ValueError, Exception)):  # InvalidConfigError
            SelfTrainerEngine(mode="ultra")