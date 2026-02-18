import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib

from core.task_detector import detect_task_type
from core.model_registry import get_models
from core.evaluator import evaluate_model
from core.threshold import find_optimal_threshold
from core.config import EngineConfig

from explainability.explainer import ModelExplainer
from versioning.model_store import save_model

from monitoring.experiment_tracker import log_experiment
from monitoring.baseline import build_baseline
from monitoring.drift_detector import detect_drift_from_baseline as detect_drift

from optimization.optuna_tuner import optimize_model


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
        print("Detected Task Type:", self.task_type)

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

        # ---------------------------
        # Drift Baseline
        # ---------------------------
        self.baseline = build_baseline(X_train)

        # ---------------------------
        # Get Candidate Models
        # ---------------------------
        models = get_models(self.task_type, y_train, self.config)
        results = {}

        print("\nCross-validation on TRAIN set:\n")

        for name, model in models.items():
            score = evaluate_model(
                model,
                X_train,
                y_train,
                self.task_type,
                self.config
            )
            results[name] = score
            print(f"{name}: {score:.6f}")

        # ---------------------------
        # Select Top 2 Models
        # ---------------------------
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_models = sorted_models[:2]
        tuned_results = {}

        # ---------------------------
        # Hyperparameter Optimization
        # ---------------------------
        for model_name, _ in top_models:

            print(f"\n🔧 Tuning {model_name} with Optuna...")

            tuned_model, best_params = optimize_model(
                model_name,
                models[model_name],
                X_train,
                y_train,
                self.task_type,
                self.config
            )

            tuned_score = evaluate_model(
                tuned_model,
                X_train,
                y_train,
                self.task_type,
                self.config
            )

            tuned_results[model_name] = {
                "model": tuned_model,
                "score": tuned_score,
                "params": best_params
            }

            print(f"Tuned {model_name} Score: {tuned_score:.6f}")

        # ---------------------------
        # Select Final Best Model
        # ---------------------------
        self.best_model_name = max(
            tuned_results,
            key=lambda x: tuned_results[x]["score"]
        )

        self.best_model = tuned_results[self.best_model_name]["model"]
        best_params = tuned_results[self.best_model_name]["params"]

        print(f"\nFinal Best Model After Tuning: {self.best_model_name}")

        # ---------------------------
        # Final Training
        # ---------------------------
        self.best_model.fit(X_train, y_train)
        self.results = results

        # ---------------------------
        # Threshold Optimization
        # ---------------------------
        if self.task_type == "classification":

            self.optimal_threshold, best_f1 = find_optimal_threshold(
                self.best_model,
                X_val,
                y_val
            )

            print(f"\nOptimal Threshold (validation): {self.optimal_threshold:.4f}")
            print(f"Validation F1: {best_f1:.4f}")

        # ---------------------------
        # Final Test Evaluation
        # ---------------------------
        print("\nFinal Test Evaluation:\n")

        test_score = evaluate_model(
            self.best_model,
            X_test,
            y_test,
            self.task_type,
            self.config
        )

        print(f"Test Score: {test_score:.6f}")

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
            "best_params": best_params
        }

        self.model_path, self.meta_path = save_model(
            self.best_model,
            metadata
        )

        print("\nModel saved to:", self.model_path)
        print("Metadata saved to:", self.meta_path)

        # ---------------------------
        # Experiment Logging
        # ---------------------------
        self.experiment_path = log_experiment(
            results=self.results,
            best_model=self.best_model_name,
            task_type=self.task_type,
            test_score=float(test_score),
            optimal_threshold=float(self.optimal_threshold),
            dataset_size=len(df)
        )

        print("Experiment logged at:", self.experiment_path)

        # ---------------------------
        # Explainability
        # ---------------------------
        self.X_train_sample = X_train.sample(
            min(self.config.shap_sample_size, len(X_train)),
            random_state=42
        )

        self.explainer = ModelExplainer(self.best_model)
        self.explainer.fit(self.X_train_sample)

    # ==========================================================
    # PREDICTION
    # ==========================================================
    def predict(self, X):

        if self.best_model is None:
            raise RuntimeError("Model not trained.")

        if self.task_type == "classification":
            probs = self.best_model.predict_proba(X)[:, 1]
            return (probs >= self.optimal_threshold).astype(int)

        return self.best_model.predict(X)

    # ==========================================================
    # DRIFT DETECTION
    # ==========================================================
    def check_drift(self, new_df):

        if self.baseline is None:
            raise RuntimeError("Baseline not available.")

        drift_report, drifted_features = detect_drift(
            self.baseline,
            new_df,
            threshold=self.config.drift_threshold
        )

        print("\nDrift Report (PSI Scores):")
        for feature, psi in drift_report.items():
            print(f"{feature}: {psi:.4f}")

        if drifted_features:
            print("\n⚠ Drifted Features Detected:")
            for feature, psi in drifted_features.items():
                print(f"{feature}: {psi:.4f}")
        else:
            print("\nNo significant drift detected.")

        return drift_report, drifted_features

    # ==========================================================
    # EXPLAINABILITY
    # ==========================================================
    def explain_global(self):

        if self.explainer is None:
            raise RuntimeError("Explainer not initialized.")

        path = self.explainer.global_explanation(
            self.X_train_sample,
            save_path="global_shap.png"
        )

        print("Global SHAP plot saved to:", path)

    def explain_instance(self, X_instance):

        if self.explainer is None:
            raise RuntimeError("Explainer not initialized.")

        contributions = self.explainer.explain_instance(X_instance)

        print("\nTop Feature Contributions:")
        for feature, value in contributions:
            direction = "increased" if value > 0 else "decreased"
            print(f"{feature} {direction} prediction impact ({value:.5f})")

    # ==========================================================
    # LOAD MODEL
    # ==========================================================
    def load(self, model_path, meta_path):

        self.best_model = joblib.load(model_path)
        metadata = joblib.load(meta_path)

        self.task_type = metadata["task_type"]
        self.best_model_name = metadata["best_model"]
        self.results = metadata["cv_results"]
        self.optimal_threshold = metadata.get("optimal_threshold", 0.5)
        self.baseline = metadata.get("baseline", None)

        self.explainer = ModelExplainer(self.best_model)

        print("Model loaded successfully.")

    # ==========================================================
    # SUMMARY
    # ==========================================================
    def summary(self):

        if self.results is None:
            raise RuntimeError("No results available.")

        print("\nModel Performance Summary")
        for name, score in self.results.items():
            print(f"{name}: {score:.6f}")
