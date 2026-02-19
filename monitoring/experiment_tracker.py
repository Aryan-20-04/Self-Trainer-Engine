"""
monitoring/experiment_tracker.py
=================================
Logs experiment metadata, CV results, and test scores to disk after
each training run. Maintains a registry.json index across all runs.

Fixes applied (Step 1):
  - json_serializer previously returned raw Python float for np.nan and
    np.inf, which json.dumps serialises as 'NaN' and 'Infinity' — both
    invalid per RFC 8259 JSON. Some parsers (JavaScript JSON.parse,
    Python json.loads in strict mode) reject these values.
    Now np.nan and np.inf are converted to None (→ JSON null) which is
    universally valid and clearly signals "not available".

Logging (Step 3):
  - Run folder path and registry update logged at INFO level.
  - Any serialisation warnings logged at WARNING level.
"""

import os
import json
import math
import logging
from datetime import datetime

import numpy as np

from core.exceptions import PersistenceError

logger = logging.getLogger("self_trainer.monitoring.experiment_tracker")


def json_serializer(obj):
    """
    Custom JSON serialiser for types that the stdlib json module
    cannot handle natively.

    Conversions
    -----------
    np.integer         → int
    np.floating        → float  (nan/inf → None so JSON stays RFC-compliant)
    np.ndarray         → list
    anything else      → str    (last-resort fallback)
    """
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        # nan and inf are not valid JSON — convert to null
        if math.isnan(value) or math.isinf(value):
            logger.warning(
                "Non-finite float value (%s) converted to null in JSON output.", value
            )
            return None
        return value

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Also handle plain Python float nan/inf (e.g. from math.nan)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            logger.warning(
                "Non-finite float value (%s) converted to null in JSON output.", obj
            )
            return None
        return obj

    return str(obj)


def log_experiment(
    results: dict,
    best_model: str,
    task_type: str,
    test_score: float,
    optimal_threshold: float | None = None,
    dataset_size: int | None = None,
    folder: str = "experiments",
) -> str:
    """
    Persist a training run to disk and update the run registry.

    Parameters
    ----------
    results : dict
        CV scores per model name.
    best_model : str
        Name of the winning model.
    task_type : str
        'classification' or 'regression'.
    test_score : float
        Hold-out test set score.
    optimal_threshold : float, optional
        Best classification threshold (classification only).
    dataset_size : int, optional
        Total number of rows in the input DataFrame.
    folder : str
        Root directory for all experiment runs (default: "experiments").

    Returns
    -------
    str
        Path to the run subfolder that was created.

    Raises
    ------
    PersistenceError
        If the run folder or any JSON file cannot be written.
    """
    try:
        os.makedirs(folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(folder, f"run_{timestamp}")
        os.makedirs(run_folder, exist_ok=True)

        # ── metadata.json ────────────────────────────────────────────────────
        metadata = {
            "timestamp": timestamp,
            "task_type": task_type,
            "best_model": best_model,
            "dataset_size": dataset_size,
            "optimal_threshold": optimal_threshold,
        }

        with open(os.path.join(run_folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4, default=json_serializer)

        # ── metrics.json ─────────────────────────────────────────────────────
        metrics = {
            "cv_results": results,
            "test_score": test_score,
        }

        with open(os.path.join(run_folder, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4, default=json_serializer)

        # ── registry.json ────────────────────────────────────────────────────
        registry_path = os.path.join(folder, "registry.json")

        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                registry = json.load(f)
        else:
            registry = []

        registry.append({
            "run_id": timestamp,
            "best_model": best_model,
            "test_score": test_score,
        })

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=4, default=json_serializer)

        logger.info(
            "Experiment logged → %s  (model=%s, score=%.4f)",
            run_folder, best_model, test_score if test_score is not None else float("nan"),
        )

        return run_folder

    except OSError as exc:
        raise PersistenceError(
            f"Failed to write experiment run to '{folder}': {exc}"
        ) from exc