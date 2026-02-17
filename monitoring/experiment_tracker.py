import os
import json
from datetime import datetime
import numpy as np

def json_serializer(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance (obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)
def log_experiment(results,
                   best_model,
                   task_type,
                   test_score,
                   optimal_threshold=None,
                   dataset_size=None,
                   folder="experiments"):

    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(folder, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    # -------------------------
    # Save metadata
    # -------------------------
    metadata = {
        "timestamp": timestamp,
        "task_type": task_type,
        "best_model": best_model,
        "dataset_size": dataset_size,
        "optimal_threshold": optimal_threshold
    }

    with open(os.path.join(run_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, default=json_serializer)

    # -------------------------
    # Save metrics
    # -------------------------
    metrics = {
        "cv_results": results,
        "test_score": test_score
    }

    with open(os.path.join(run_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4, default=json_serializer)

    # -------------------------
    # Update registry
    # -------------------------
    registry_path = os.path.join(folder, "registry.json")

    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = []

    registry.append({
        "run_id": timestamp,
        "best_model": best_model,
        "test_score": test_score
    })

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4, default=json_serializer)

    return run_folder
