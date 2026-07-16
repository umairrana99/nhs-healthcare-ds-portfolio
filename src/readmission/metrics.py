"""Calibration metrics.

Expected Calibration Error (ECE) complements the Brier score: it bins predictions
by confidence and averages the gap between predicted probability and observed
frequency, so a well-discriminating but over-confident model is still penalised.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def expected_calibration_error(
    y_true: npt.NDArray[Any],
    y_prob: npt.NDArray[Any],
    *,
    n_bins: int = 10,
) -> float:
    """Return the ECE: |accuracy - confidence| averaged over probability bins."""
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    total = len(prob)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(prob[mask].mean())
        accuracy = float(truth[mask].mean())
        ece += (count / total) * abs(accuracy - confidence)
    return ece


def net_benefit(
    y_true: npt.NDArray[Any],
    y_prob: npt.NDArray[Any],
    *,
    thresholds: npt.NDArray[Any] | None = None,
) -> dict[str, npt.NDArray[Any]]:
    """Decision-curve net benefit across probability thresholds.

    ``NB(pt) = TP/n - FP/n * pt/(1-pt)`` — weighs true positives against false
    positives at the clinician's threshold probability, answering "is acting on
    this model better than treat-all / treat-none?".
    """
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    grid = np.linspace(0.01, 0.99, 99) if thresholds is None else np.asarray(thresholds, float)
    total = len(truth)
    benefits = np.empty(len(grid))
    for i, threshold in enumerate(grid):
        predicted = prob >= threshold
        tp = float(np.sum(predicted & (truth == 1)))
        fp = float(np.sum(predicted & (truth == 0)))
        benefits[i] = tp / total - fp / total * (threshold / (1.0 - threshold))
    return {"threshold": grid, "net_benefit": benefits}
