import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import numpy as np


def optimize_model(model_name, base_model, X, y, task_type, config):

    def objective(trial):

        model = clone(base_model)

        # -----------------------------
        # Parameter Suggestions
        # -----------------------------
        if model_name == "XGBoost":
            model.set_params(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            )

        elif model_name == "RandomForest":
            model.set_params(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 20),
            )

        elif model_name == "LightGBM":
            model.set_params(
                n_estimators=trial.suggest_int("n_estimators", 30, 200),
                max_depth=trial.suggest_int("max_depth", -1, 15),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            )

        # -----------------------------
        # Metric Selection
        # -----------------------------
        scoring = (
            config.classification_metric
            if task_type == "classification"
            else config.regression_metric
        )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=config.cv_folds,
            scoring=scoring,
            n_jobs=1
        )

        return float(np.mean(scores))

    direction = "maximize" if task_type == "classification" else "minimize"

    study = optuna.create_study(
        direction=direction,
        pruner=MedianPruner(n_warmup_steps=2)
    )

    study.optimize(
        objective,
        timeout=config.optuna_timeout,
        show_progress_bar=False
    )

    # Apply best params to fresh model
    tuned_model = clone(base_model)
    tuned_model.set_params(**study.best_params)

    print(f"\nBest params for {model_name}: {study.best_params}")
    print(f"Best CV score after tuning: {study.best_value:.6f}")

    return tuned_model, study.best_params
