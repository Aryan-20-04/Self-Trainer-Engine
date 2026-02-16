from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from typing import Dict,Any

def detect_imbalance(y):

    class_distribution = y.value_counts(normalize=True)

    if len(class_distribution) == 2 and class_distribution.min() < 0.1:
        return True

    return False

def get_models(task_type, y) -> Dict[str, Any]:

    imbalance = detect_imbalance(y)

    if task_type == "classification":

        return {
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    solver="saga",
                    class_weight="balanced" if imbalance else None
                ))
            ]),

            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced" if imbalance else None,
                n_jobs=-1
            ),

            "XGBoost": XGBClassifier(
                n_estimators=100,
                scale_pos_weight=(
                    (y == 0).sum() / (y == 1).sum()
                    if imbalance else 1
                ),
                eval_metric="logloss",
                n_jobs=-1
            ),

            "LightGBM": LGBMClassifier(
                n_estimators=100,
                class_weight="balanced" if imbalance else None,
                n_jobs=-1,
                force_col_wise=True
            ),

            "MLP": MLPClassifier(
                max_iter=500
            )
        }

    elif task_type == "regression":

        return {
            "LinearRegression": LinearRegression(),

            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                n_jobs=-1
            ),

            "XGBoost": XGBRegressor(
                n_estimators=100,
                n_jobs=-1
            ),

            "LightGBM": LGBMRegressor(
                n_estimators=100,
                n_jobs=-1,
                force_col_wise=True
            ),

            "MLP": MLPRegressor(max_iter=500)
        }

    else:
        raise ValueError("Invalid task type")
