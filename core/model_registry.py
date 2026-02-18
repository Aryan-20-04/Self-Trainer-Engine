from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from typing import Dict, Any


# -------------------------------------------------
# Imbalance Detection
# -------------------------------------------------
def detect_imbalance(y) -> bool:
    class_distribution = y.value_counts(normalize=True)

    # Binary classification only
    if len(class_distribution) == 2 and class_distribution.min() < 0.15:
        return True

    return False


# -------------------------------------------------
# Model Registry
# -------------------------------------------------
def get_models(task_type, y, config) -> Dict[str, Any]:

    imbalance = detect_imbalance(y)

    if task_type == "classification":

        scale_pos_weight = (
            (y == 0).sum() / (y == 1).sum()
            if imbalance else 1
        )

        models = {

            # -----------------------------
            # Logistic Regression
            # -----------------------------
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=3000,
                    solver="saga",
                    penalty="l2",
                    class_weight="balanced" if imbalance else None,
                    n_jobs=-1
                ))
            ]),

            # -----------------------------
            # Random Forest
            # -----------------------------
            "RandomForest": RandomForestClassifier(
                n_estimators=config.tree_estimators,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced" if imbalance else None,
                n_jobs=-1,
                random_state=42
            ),

            # -----------------------------
            # XGBoost (Usually strongest)
            # -----------------------------
            "XGBoost": XGBClassifier(
                n_estimators=config.tree_estimators,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42
            ),

            # -----------------------------
            # LightGBM (Stable version)
            # -----------------------------
            "LightGBM": LGBMClassifier(
                n_estimators=config.tree_estimators,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                class_weight="balanced" if imbalance else None,
                force_col_wise=True,
                verbose=-1,              # 🔥 Removes warning spam
                n_jobs=-1,
                random_state=42
            ),

            # -----------------------------
            # MLP
            # -----------------------------
            "MLP": Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=800,
                    random_state=42
                ))
            ])
        }

        # Optional: disable LightGBM in dev mode for small datasets
        if config.mode == "dev" and len(y) < 10000:
            models.pop("LightGBM", None)

        return models

    # -------------------------------------------------
    # Regression
    # -------------------------------------------------
    elif task_type == "regression":

        return {

            "LinearRegression": LinearRegression(),

            "RandomForest": RandomForestRegressor(
                n_estimators=config.tree_estimators,
                max_depth=None,
                n_jobs=-1,
                random_state=42
            ),

            "XGBoost": XGBRegressor(
                n_estimators=config.tree_estimators,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                n_jobs=-1,
                random_state=42
            ),

            "LightGBM": LGBMRegressor(
                n_estimators=config.tree_estimators,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                force_col_wise=True,
                verbose=-1,
                n_jobs=-1,
                random_state=42
            ),

            "MLP": Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=800,
                    random_state=42
                ))
            ])
        }

    else:
        raise ValueError("Invalid task type")
