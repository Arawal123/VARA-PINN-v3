"""Residual-weighted point resampling."""

from __future__ import annotations

import numpy as np


def sample_from_score_grid(
    coords: np.ndarray,
    score: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample coordinates from an empirical score distribution."""
    if n <= 0:
        return np.zeros((0, coords.shape[1]), dtype=float)
    flat_score = np.asarray(score, dtype=float).reshape(-1)
    flat_score = np.where(np.isfinite(flat_score), np.maximum(flat_score, 0.0), 0.0)
    flat_score = flat_score + 1e-12
    prob = flat_score / flat_score.sum()
    idx = rng.choice(coords.shape[0], size=n, replace=True, p=prob)
    return coords[idx]
