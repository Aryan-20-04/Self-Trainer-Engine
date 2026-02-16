import pandas as pd
from sklearn.model_selection import train_test_split

from core.task_detector import detect_task_type
from core.model_registry import get_models
from core.evaluator import evaluate_model
from core.threshold import find_optimal_threshold

from explainability.explainer import ModelExplainer


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
        # Explainability (use train sample)
        # -------------------------------
        self.X_train_sample = X_train.sample(
            min(1000, len(X_train)),
            random_state=42
        )

        self.explainer = ModelExplainer(self.best_model)
        self.explainer.fit(self.X_train_sample)
    
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
