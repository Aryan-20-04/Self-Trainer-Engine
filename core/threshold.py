import numpy as np
from sklearn.metrics import precision_recall_curve


def find_optimal_threshold(model, X, y):

    probs = model.predict_proba(X)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, probs)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)

    # thresholds array is shorter by 1 than precision/recall
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]

    return best_threshold, f1_scores[best_idx]
