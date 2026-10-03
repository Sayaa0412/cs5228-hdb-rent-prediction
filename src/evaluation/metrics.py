"""Validate regression outputs and calculate project metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def validate_predictions(prediction, expected_rows: int, *, positive: bool = False) -> np.ndarray:
    values = np.asarray(prediction, dtype=float)
    if values.shape != (expected_rows,):
        raise ValueError(f"Expected {expected_rows} scalar predictions, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Predictions must be finite")
    if positive and (values <= 0).any():
        raise ValueError("Rent predictions must be positive")
    return values


def regression_metrics(y_true, prediction) -> dict[str, float]:
    target = np.asarray(y_true, dtype=float)
    if target.ndim != 1 or len(target) == 0 or not np.isfinite(target).all():
        raise ValueError("Targets must be a nonempty, finite one-dimensional array")
    values = validate_predictions(prediction, len(target))
    return {
        "mae_sgd": round(float(mean_absolute_error(target, values)), 2),
        "rmse_sgd": round(float(np.sqrt(mean_squared_error(target, values))), 2),
        "r2": round(float(r2_score(target, values)), 4),
    }
