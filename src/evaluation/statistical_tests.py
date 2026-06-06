"""Small statistical helpers for multi-seed comparisons."""

from __future__ import annotations

import math
import numpy as np


def paired_mean_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a) - np.asarray(b)
    return {
        "mean_difference": float(np.mean(diff)),
        "std_difference": float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0,
        "n": int(diff.size),
    }


def paired_bootstrap_improvement(
    baseline: np.ndarray,
    method: np.ndarray,
    samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap interval for lower-is-better percentage improvement."""
    baseline = np.asarray(baseline, dtype=float)
    method = np.asarray(method, dtype=float)
    mask = np.isfinite(baseline) & np.isfinite(method) & (np.abs(baseline) > 1e-12)
    baseline = baseline[mask]
    method = method[mask]
    if baseline.size == 0:
        return {"mean_improvement_percent": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    paired = 100.0 * (baseline - method) / np.abs(baseline)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, paired.size, size=(int(samples), paired.size))
    means = paired[indices].mean(axis=1)
    alpha = 1.0 - float(confidence)
    return {
        "mean_improvement_percent": float(paired.mean()),
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "n": int(paired.size),
    }


def wilcoxon_signed_rank(baseline: np.ndarray, method: np.ndarray) -> dict[str, float]:
    """Two-sided paired Wilcoxon signed-rank test without SciPy."""
    difference = np.asarray(method, dtype=float) - np.asarray(baseline, dtype=float)
    difference = difference[np.isfinite(difference)]
    difference = difference[np.abs(difference) > 1e-14]
    n = int(difference.size)
    if n == 0:
        return {"wilcoxon_statistic": 0.0, "p_value": 1.0, "n": 0}
    ranks = _average_ranks(np.abs(difference))
    positive = float(np.sum(ranks[difference > 0]))
    negative = float(np.sum(ranks[difference < 0]))
    statistic = min(positive, negative)
    if n <= 20:
        totals = np.array([0.0])
        for rank in ranks:
            totals = np.concatenate([totals, totals + rank])
        probability = float(np.mean((totals <= statistic + 1e-12) | (totals >= ranks.sum() - statistic - 1e-12)))
        p_value = min(1.0, probability)
    else:
        mean = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0
        z = (statistic - mean + 0.5) / np.sqrt(max(variance, 1e-12))
        p_value = float(math.erfc(abs(z) / np.sqrt(2.0)))
    return {"wilcoxon_statistic": statistic, "p_value": p_value, "n": n}


def paired_effect_size(baseline: np.ndarray, method: np.ndarray) -> float:
    """Paired Cohen dz; positive means the method is better for lower-is-better metrics."""
    difference = np.asarray(baseline, dtype=float) - np.asarray(method, dtype=float)
    difference = difference[np.isfinite(difference)]
    if difference.size < 2:
        return float("nan")
    std = float(np.std(difference, ddof=1))
    return float(np.mean(difference) / std) if std > 1e-12 else float("inf")


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average
        start = end
    return ranks
