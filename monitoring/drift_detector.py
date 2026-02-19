"""
monitoring/drift_detector.py
============================
Computes Population Stability Index (PSI) between a stored baseline
and new incoming data to detect feature distribution drift.

Fixes applied (Step 1):
  - Out-of-range new data (entirely outside baseline bin edges) previously
    produced actual_counts.sum() == 0 → division by zero → silent NaN PSI.
    Now raises DriftCalculationError with a clear message.
  - Empty new data (0 rows) is caught before histogram computation.
  - NaN-only new data is caught after dropping NaN.
  - drifted_features filtering skips NaN PSI values so they never silently
    appear in the drift report.

Structured errors (Step 2):
  - DriftCalculationError replaces RuntimeWarning + NaN return.

Logging (Step 3):
  - Per-feature PSI values logged at DEBUG level.
  - Drifted features logged at WARNING level.
  - Calculation failures logged at ERROR level (column skipped, not crashed).
"""

import logging
import numpy as np
import pandas as pd

from core.exceptions import DriftCalculationError

logger = logging.getLogger("self_trainer.monitoring.drift_detector")


def calculate_psi_from_baseline(baseline_info: dict, actual_values: pd.Series) -> float:
    """
    Compute PSI for one feature against its stored baseline histogram.

    Parameters
    ----------
    baseline_info : dict
        {"bin_edges": [...], "percentages": [...]} from build_baseline.
    actual_values : pd.Series
        New data for this feature.

    Returns
    -------
    float
        PSI score (0 = no drift, >0.2 = significant drift).

    Raises
    ------
    DriftCalculationError
        If actual_values is empty, all-NaN, or entirely outside the
        baseline bin edges (would produce a meaningless NaN PSI).
    """
    bin_edges = np.array(baseline_info["bin_edges"])
    expected_perc = np.array(baseline_info["percentages"])

    # ── Sanitise new data ────────────────────────────────────────────────────
    arr = actual_values.to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]  # drop NaN

    if len(arr) == 0:
        raise DriftCalculationError(
            "Cannot compute PSI: new data is empty or all-NaN after dropping missing values."
        )

    # ── Histogram against baseline edges ────────────────────────────────────
    actual_counts, _ = np.histogram(arr, bins=bin_edges)
    total = actual_counts.sum()

    if total == 0:
        raise DriftCalculationError(
            "Cannot compute PSI: all new data values fall outside the baseline bin edges. "
            "The new data distribution may be on a completely different scale. "
            f"Baseline range: [{bin_edges[0]:.4g}, {bin_edges[-1]:.4g}], "
            f"new data range: [{arr.min():.4g}, {arr.max():.4g}]."
        )

    actual_perc = actual_counts / total

    # ── PSI formula with epsilon to avoid log(0) ─────────────────────────────
    eps = 1e-8
    psi = float(np.sum(
        (actual_perc - expected_perc) *
        np.log((actual_perc + eps) / (expected_perc + eps))
    ))

    return psi


def detect_drift_from_baseline(
    baseline: dict,
    new_df: pd.DataFrame,
    threshold: float,
) -> tuple[dict, dict]:
    """
    Compute PSI for every feature present in both the baseline and new_df.

    Parameters
    ----------
    baseline : dict
        Output of build_baseline().
    new_df : pd.DataFrame
        New incoming data to check for drift.
    threshold : float
        Features with PSI above this value are flagged as drifted.

    Returns
    -------
    drift_report : dict
        { feature: psi_score } — only features that could be computed.
    drifted_features : dict
        Subset of drift_report where psi_score > threshold.
    """
    drift_report = {}

    for col, baseline_info in baseline.items():

        if col not in new_df.columns:
            logger.debug("Column '%s' not in new data — skipped.", col)
            continue

        try:
            psi = calculate_psi_from_baseline(baseline_info, new_df[col])
            drift_report[col] = psi
            logger.debug("PSI for '%s': %.4f", col, psi)

        except DriftCalculationError as exc:
            logger.error(
                "PSI calculation failed for column '%s': %s — column skipped.", col, exc
            )
            # Column is excluded from the report rather than producing a NaN entry

    drifted_features = {
        k: v for k, v in drift_report.items()
        if np.isfinite(v) and v > threshold
    }

    if drifted_features:
        logger.warning(
            "Drift detected in %d feature(s): %s",
            len(drifted_features),
            list(drifted_features.keys()),
        )
    else:
        logger.info("No significant drift detected (threshold=%.3f).", threshold)

    return drift_report, drifted_features