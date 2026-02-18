import numpy as np
import pandas as pd


def calculate_psi_from_baseline(baseline_info, actual_values):

    bin_edges = np.array(baseline_info["bin_edges"])
    expected_perc = np.array(baseline_info["percentages"])

    actual_counts, _ = np.histogram(actual_values, bins=bin_edges)
    actual_perc = actual_counts / actual_counts.sum()

    psi = np.sum(
        (actual_perc - expected_perc) *
        np.log((actual_perc + 1e-8) / (expected_perc + 1e-8))
    )

    return psi


def detect_drift_from_baseline(baseline, new_df, threshold):

    drift_report = {}

    for col, baseline_info in baseline.items():

        if col in new_df.columns:

            psi = calculate_psi_from_baseline(
                baseline_info,
                new_df[col]
            )

            drift_report[col] = psi

    drifted_features = {
        k: v for k, v in drift_report.items()
        if v > threshold
    }

    return drift_report, drifted_features
