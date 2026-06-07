"""Severity score normalization."""

from __future__ import annotations

import numpy as np


def normalize_values(values: np.ndarray, method: str = "percentile", eps: float = 1e-12) -> np.ndarray:
    """Normalize a nonnegative array for severity comparisons."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    method = method.lower()
    if method == "percentile":
        scale = np.nanpercentile(arr, 95)
        return arr / (scale + eps)
    if method == "zscore":
        return (arr - np.nanmean(arr)) / (np.nanstd(arr) + eps)
    if method == "median_iqr":
        med = np.nanmedian(arr)
        q75, q25 = np.nanpercentile(arr, [75, 25])
        return (arr - med) / (q75 - q25 + eps)
    if method == "minmax":
        return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + eps)
    if method == "none":
        return arr
    raise ValueError(f"Unknown normalization method: {method}")


def robust_aggregate(values: np.ndarray, mode: str = "mean", percentile: float = 90.0) -> float:
    """Aggregate point values into one patch severity."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    mode = mode.lower()
    if mode == "mean":
        return float(np.mean(values))
    if mode == "max":
        return float(np.max(values))
    if mode == "percentile":
        return float(np.percentile(values, percentile))
    if mode == "robust_mean":
        hi = np.percentile(values, 90)
        clipped = np.clip(values, None, hi)
        return float(np.mean(clipped))
    raise ValueError(f"Unknown aggregation mode: {mode}")
