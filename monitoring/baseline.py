"""
monitoring/baseline.py
======================
Builds a histogram-based statistical baseline from training data.
Used later by drift_detector to compute PSI scores.

Fixes applied (Step 1):
  - NaN values in a column are dropped before np.histogram is called.
    Previously raised: ValueError: autodetected range of [nan, nan] is not finite
  - Inf / -Inf values are replaced with the column's finite min/max before
    histogramming. Previously raised: ValueError: autodetected range not finite
  - Columns where all values are NaN or Inf (no finite data at all) are
    skipped and logged as a warning rather than crashing the whole pipeline.
  - Very large value ranges (e.g. 1e308) that cause np.histogram to error
    with "Too many bins" are caught and the column is skipped with a warning.

Logging (Step 3):
  - All skip/warning events are emitted through the module logger so they
    appear in the application log rather than silently disappearing.
"""

import logging
import numpy as np
import pandas as pd

from core.exceptions import InvalidDatasetError

logger = logging.getLogger("self_trainer.monitoring.baseline")


def _sanitise_column(series: pd.Series) -> np.ndarray | None:
    """
    Return a finite numpy array suitable for np.histogram, or None if the
    column has no usable data.

    Steps:
      1. Drop NaN.
      2. Replace +/-Inf with the column's finite max/min.
      3. Return None if no finite values remain.
    """
    arr = series.to_numpy(dtype=float)

    # Drop NaN first
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None

    # Replace Inf with finite boundary
    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        finite_vals = arr[finite_mask]
        if len(finite_vals) == 0:
            return None
        arr = np.clip(arr, finite_vals.min(), finite_vals.max())

    return arr


def build_baseline(reference_df: pd.DataFrame, bins: int = 10) -> dict:
    """
    Build a per-column histogram baseline from a reference DataFrame.

    Parameters
    ----------
    reference_df : pd.DataFrame
        The training data to baseline against (numeric columns only).
    bins : int
        Number of histogram bins (default 10).

    Returns
    -------
    dict
        { column_name: { "bin_edges": [...], "percentages": [...] } }
        Only numeric columns with at least one finite value are included.

    Raises
    ------
    InvalidDatasetError
        If reference_df is not a DataFrame.
    """
    if not isinstance(reference_df, pd.DataFrame):
        raise InvalidDatasetError(
            f"build_baseline expects a DataFrame, got {type(reference_df).__name__}"
        )

    baseline = {}

    for col in reference_df.columns:
        if not pd.api.types.is_numeric_dtype(reference_df[col]):
            continue

        arr = _sanitise_column(reference_df[col])

        if arr is None:
            logger.warning(
                "Column '%s' has no finite values after sanitisation — skipped.", col
            )
            continue

        try:
            counts, bin_edges = np.histogram(arr, bins=bins)
        except ValueError as exc:
            logger.warning(
                "Column '%s' could not be histogrammed (%s) — skipped.", col, exc
            )
            continue

        total = counts.sum()
        if total == 0:
            logger.warning(
                "Column '%s' produced an empty histogram — skipped.", col
            )
            continue

        percentages = counts / total

        baseline[col] = {
            "bin_edges": bin_edges.tolist(),
            "percentages": percentages.tolist(),
        }

    logger.debug("Baseline built for %d column(s).", len(baseline))
    return baseline