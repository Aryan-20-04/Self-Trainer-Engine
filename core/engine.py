import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib

from core.task_detector import detect_task_type
from core.model_registry import get_models
from core.evaluator import evaluate_model
from core.threshold import find_optimal_threshold

from explainability.explainer import ModelExplainer
from versioning.model_store import save_model
from monitoring.experiment_tracker import log_experiment

from monitoring.drift_detector import detect_drift
class SelfTrainerEngine:

    def __init__(self):
        self.task_type = None
        self.best_model = None
        self.best_model_name = None
        self.results = None
        self.explainer = None
        self.X_train_sample = None
        self.optimal_threshold = 0.5  # default

    def fit(self, df, target, test_size=0.2, val_size=0.2):

        X = df.drop(target, axis=1)
        y = df[target]

        self.task_type = detect_task_type(y)

        print("Detected Task Type:", self.task_type)

        # -------------------------------
        # First split: Train+Val vs Test
        # -------------------------------
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # -------------------------------
        # Second split: Train vs Validation
        # -------------------------------
        val_relative_size = val_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative_size,
            random_state=42
        )
        
        self.reference_data = X_train.copy()

        models = get_models(self.task_type, y_train)

        results = {}

        print("\nCross-validation on TRAIN set:\n")

        for name, model in models.items():
            score = evaluate_model(model, X_train, y_train, self.task_type)
            results[name] = score
            print(f"{name}: {score:.6f}")

        self.best_model_name = max(results, key=results.get) #type: ignore
        self.best_model = models[self.best_model_name]

        # Final training on TRAIN set
        self.best_model.fit(X_train, y_train)
        self.results = results

        print("\nBest Model:", self.best_model_name)

        # -------------------------------
        # Threshold Optimization (Validation only)
        # -------------------------------
        if self.task_type == "classification":
  
            self.optimal_threshold, best_f1 = find_optimal_threshold(
                self.best_model,
                X_val,
                y_val
            )

            print(f"\nOptimal Threshold (validation): {self.optimal_threshold:.4f}")
            print(f"Validation F1 at threshold: {best_f1:.4f}")

        # -------------------------------
        # Final Test Performance
        # -------------------------------
        print("\nFinal Test Evaluation:\n")

        test_score = evaluate_model(
            self.best_model,
            X_test,
            y_test,
            self.task_type
        )

        print(f"Test Score: {test_score:.6f}")
        
        # -------------------------------
        # Model Persistence (Versioning)
        # -------------------------------
        metadata = {
            "timestamp":datetime.now().isoformat(),
            "task_type":self.task_type,
            "best_model":self.best_model_name,
            "cv_results":self.results,
            "optimal_threshold":self.optimal_threshold,
            "test_score":test_score,
            "feature_name":list(X.columns)
        }
        
        self.model_path, self.meta_path = save_model(
            self.best_model,
            metadata
        )
        
        print("\nModel saved to:", self.model_path)
        print("Metadata saved to:", self.meta_path)
        
        # -------------------------------
        # Experiment Tracking
        # -------------------------------

        self.experiment_path = log_experiment(
            results = self.results,
            best_model=self.best_model_name,
            task_type=self.task_type,
            test_score=test_score,
            optimal_threshold=self.optimal_threshold,
            dataset_size=len(df)
        )
        
        print("Experiment logged at:", self.experiment_path)

        # -------------------------------
        # Explainability (use train sample)
        # -------------------------------
        self.X_train_sample = X_train.sample(
            min(1000, len(X_train)),
            random_state=42
        )

        self.explainer = ModelExplainer(self.best_model)
        self.explainer.fit(self.X_train_sample)
        
    # ===============================
    # Drift Detection
    # ===============================
    
    def check_drift(self, new_df):
        
        if not hasattr(self, "reference_data"):
            raise RuntimeError("No reference data available. Call fit() first.")
        
        drift_report, drifted_features = detect_drift(
            self.reference_data,
            new_df
        )
        
        print("\nDrift Report(PSI Scores):")
        for features, psi in drift_report.items():
            print(f"{features}: {psi:.4f}")
            
        if drifted_features:
            print("\nDrifted Features:")
            for feature, psi in drifted_features.items():
                print(f"{feature}: {psi:.4f}")
        else:
            print("\nNo significant drift detected.")
            
        return drift_report, drifted_features
    
    # ===============================
    # Prediction
    # ===============================

    def predict(self, X):

        if self.best_model is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        if self.task_type == "classification":

            probs = self.best_model.predict_proba(X)[:, 1]
            return (probs >= self.optimal_threshold).astype(int)

        return self.best_model.predict(X)
    
    def load(self, model_path, meta_path):
        self.best_model = joblib.load(model_path)
        metadata = joblib.load(meta_path)
        
        self.task_type = metadata["task_type"]
        self.best_model_name = metadata["best_model"]
        self.results = metadata["cv_results"]
        self.optimal_threshold = metadata.get("optimal_threshold", 0.5)
        
        self.explainer = ModelExplainer(self.best_model)
        
        print("Model loaded successfully")

    # ===============================
    # Explainability
    # ===============================

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

    # ===============================
    # Summary
    # ===============================

    def summary(self):

        if self.results is None:
            raise RuntimeError("No training results available. Call fit() first.")

        print("\nModel Performance Summary")
        for name, score in self.results.items():
            print(f"{name}: {score:.6f}")
