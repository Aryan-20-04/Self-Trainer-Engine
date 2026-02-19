"""
core/engine.py
==============
SelfTrainerEngine — the main orchestration class.

Changes (Steps 2 & 3):
  - All print() calls replaced with logger calls at appropriate levels.
  - All bare RuntimeError raises replaced with structured exception classes.
  - A single module-level logger is used throughout; callers configure it.
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib

from core.task_detector import detect_task_type
from core.model_registry import get_models
from core.evaluator import evaluate_model
from core.threshold import find_optimal_threshold
from core.config import EngineConfig
from core.exceptions import (
    ModelNotTrainedError,
    ExplainerNotReadyError,
    BaselineNotAvailableError,
    PersistenceError,
)

from explainability.explainer import ModelExplainer
from versioning.model_store import save_model

from monitoring.experiment_tracker import log_experiment
from monitoring.baseline import build_baseline
from monitoring.drift_detector import detect_drift_from_baseline as detect_drift

from optimization.optuna_tuner import optimize_model

logger = logging.getLogger("self_trainer.engine")


class SelfTrainerEngine:

    def __init__(self, mode: str = "dev"):
        self.config = EngineConfig(mode)
        self.task_type = None
        self.best_model = None
        self.best_model_name = None
        self.results = None
        self.explainer = None
        self.X_train_sample = None
        self.optimal_threshold = 0.5
        self.baseline = None
        self.model_path = None
        self.meta_path = None
        self.experiment_path = None

    # ==========================================================
    # FIT PIPELINE
    # ==========================================================
    def fit(self, df: pd.DataFrame, target: str,
            test_size: float = 0.2,
            val_size: float = 0.2):

        X = df.drop(target, axis=1)
        y = df[target]

        self.task_type = detect_task_type(y)
        logger.info("Detected task type: %s", self.task_type)

        # ---------------------------
        # Train / Validation / Test Split
        # ---------------------------
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        val_relative_size = val_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative_size,
            random_state=42
        )

        logger.info(
            "Data split — train: %d, val: %d, test: %d rows.",
            len(X_train), len(X_val), len(X_test)
        )

        # ---------------------------
        # Drift Baseline
        # ---------------------------
        self.baseline = build_baseline(X_train)
        logger.debug("Drift baseline built for %d feature(s).", len(self.baseline))

        # ---------------------------
        # Get Candidate Models
        # ---------------------------
        models = get_models(self.task_type, y_train, self.config)
        results = {}

        logger.info("Cross-validation on TRAIN set (%d folds):", self.config.cv_folds)

        for name, model in models.items():
            score = evaluate_model(
                model, X_train, y_train, self.task_type, self.config
            )
            results[name] = score
            logger.info("  %-20s %.6f", name, score)

        # ---------------------------
        # Select Top 2 Models
        # ---------------------------
        sorted_models = sorted(results.items(), key=lambda x: x[1], reverse=True)
        top_models = sorted_models[:2]
        tuned_results = {}

        # ---------------------------
        # Hyperparameter Optimisation
        # ---------------------------
        for model_name, _ in top_models:

            logger.info("Tuning %s with Optuna...", model_name)

            tuned_model, best_params = optimize_model(
                model_name, models[model_name],
                X_train, y_train,
                self.task_type, self.config
            )

            tuned_score = evaluate_model(
                tuned_model, X_train, y_train, self.task_type, self.config
            )

            tuned_results[model_name] = {
                "model": tuned_model,
                "score": tuned_score,
                "params": best_params,
            }

            logger.info("Tuned %s score: %.6f", model_name, tuned_score)

        # ---------------------------
        # Select Final Best Model
        # ---------------------------
        self.best_model_name = max(
            tuned_results, key=lambda x: tuned_results[x]["score"]
        )
        self.best_model = tuned_results[self.best_model_name]["model"]
        best_params = tuned_results[self.best_model_name]["params"]

        logger.info("Final best model after tuning: %s", self.best_model_name)

        # ---------------------------
        # Final Training
        # ---------------------------
        self.best_model.fit(X_train, y_train)
        self.results = results

        # ---------------------------
        # Threshold Optimisation
        # ---------------------------
        if self.task_type == "classification":
            self.optimal_threshold, best_f1 = find_optimal_threshold(
                self.best_model, X_val, y_val
            )
            logger.info(
                "Optimal threshold (validation): %.4f  |  F1: %.4f",
                self.optimal_threshold, best_f1
            )

        # ---------------------------
        # Final Test Evaluation
        # ---------------------------
        test_score = evaluate_model(
            self.best_model, X_test, y_test, self.task_type, self.config
        )
        logger.info("Test score: %.6f", test_score)

        # ---------------------------
        # Model Persistence
        # ---------------------------
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "task_type": self.task_type,
            "best_model": self.best_model_name,
            "cv_results": self.results,
            "optimal_threshold": float(self.optimal_threshold),
            "test_score": float(test_score),
            "baseline": self.baseline,
            "best_params": best_params,
        }

        self.model_path, self.meta_path = save_model(self.best_model, metadata)

        logger.info("Model saved → %s", self.model_path)
        logger.info("Metadata saved → %s", self.meta_path)

        # ---------------------------
        # Experiment Logging
        # ---------------------------
        self.experiment_path = log_experiment(
            results=self.results,
            best_model=self.best_model_name,
            task_type=self.task_type,
            test_score=float(test_score),
            optimal_threshold=float(self.optimal_threshold),
            dataset_size=len(df),
        )

        logger.info("Experiment logged → %s", self.experiment_path)

        # ---------------------------
        # Explainability
        # ---------------------------
        self.X_train_sample = X_train.sample(
            min(self.config.shap_sample_size, len(X_train)),
            random_state=42
        )

        self.explainer = ModelExplainer(self.best_model)
        self.explainer.fit(self.X_train_sample)
        logger.info("SHAP explainer fitted on %d samples.", len(self.X_train_sample))

    # ==========================================================
    # PREDICTION
    # ==========================================================
    def predict(self, X):

        if self.best_model is None:
            raise ModelNotTrainedError(
                "Cannot predict: model has not been trained. Call fit() first."
            )

        if self.task_type == "classification":
            probs = self.best_model.predict_proba(X)[:, 1]
            preds = (probs >= self.optimal_threshold).astype(int)
            logger.debug(
                "Predicted %d samples (threshold=%.4f).", len(preds), self.optimal_threshold
            )
            return preds

        preds = self.best_model.predict(X)
        logger.debug("Predicted %d samples (regression).", len(preds))
        return preds

    # ==========================================================
    # DRIFT DETECTION
    # ==========================================================
    def check_drift(self, new_df):

        if self.baseline is None:
            raise BaselineNotAvailableError(
                "Cannot check drift: baseline not available. Call fit() first."
            )

        drift_report, drifted_features = detect_drift(
            self.baseline,
            new_df,
            threshold=self.config.drift_threshold,
        )

        logger.info("Drift report (PSI scores):")
        for feature, psi in drift_report.items():
            logger.info("  %-20s %.4f", feature, psi)

        if drifted_features:
            logger.warning(
                "Drifted features detected: %s", list(drifted_features.keys())
            )
        else:
            logger.info("No significant drift detected.")

        return drift_report, drifted_features

    # ==========================================================
    # EXPLAINABILITY
    # ==========================================================
    def explain_global(self):

        if self.explainer is None:
            raise ExplainerNotReadyError(
                "Cannot explain: SHAP explainer not initialised. Call fit() first."
            )

        path = self.explainer.global_explanation(
            self.X_train_sample,
            save_path="global_shap.png"
        )
        logger.info("Global SHAP plot saved → %s", path)

    def explain_instance(self, X_instance):

        if self.explainer is None:
            raise ExplainerNotReadyError(
                "Cannot explain: SHAP explainer not initialised. Call fit() first."
            )

        contributions = self.explainer.explain_instance(X_instance)

        logger.info("Top feature contributions:")
        for feature, value in contributions:
            direction = "increased" if value > 0 else "decreased"
            logger.info("  %s %s prediction impact (%.5f)", feature, direction, value)

    # ==========================================================
    # LOAD MODEL
    # ==========================================================
    def load(self, model_path: str, meta_path: str):

        try:
            self.best_model = joblib.load(model_path)
            metadata = joblib.load(meta_path)
        except (FileNotFoundError, Exception) as exc:
            raise PersistenceError(
                f"Failed to load model from '{model_path}': {exc}"
            ) from exc

        self.task_type = metadata["task_type"]
        self.best_model_name = metadata["best_model"]
        self.results = metadata["cv_results"]
        self.optimal_threshold = metadata.get("optimal_threshold", 0.5)
        self.baseline = metadata.get("baseline", None)

        self.explainer = ModelExplainer(self.best_model)

        logger.info(
            "Model loaded — type=%s, model=%s, threshold=%.4f",
            self.task_type, self.best_model_name, self.optimal_threshold
        )

    # ==========================================================
    # SUMMARY
    # ==========================================================
    def summary(self):

        if self.results is None:
            raise ModelNotTrainedError(
                "No results available. Call fit() before summary()."
            )

        logger.info("Model performance summary:")
        for name, score in self.results.items():
            logger.info("  %-20s %.6f", name, score)