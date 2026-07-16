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
