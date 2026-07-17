"""Data-drift monitoring.

The Population Stability Index (PSI) compares a live feature/score distribution
against the training reference. Rule of thumb: < 0.1 no significant shift,
0.1–0.25 moderate, > 0.25 major shift (retrain / investigate).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

PSI_MODERATE = 0.1
PSI_MAJOR = 0.25


def population_stability_index(
    reference: npt.NDArray[Any],
    current: npt.NDArray[Any],
    *,
    n_bins: int = 10,
) -> float:
    """Return the PSI between a reference and current distribution."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.percentile(ref, np.linspace(0, 100, n_bins + 1)))
    if edges.size < 2:  # constant reference — no bins to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_frac = np.clip(np.histogram(ref, bins=edges)[0] / ref.size, 1e-6, None)
    cur_frac = np.clip(np.histogram(cur, bins=edges)[0] / cur.size, 1e-6, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def drift_label(psi: float) -> str:
    """Human-readable drift severity for a PSI value."""
    if psi < PSI_MODERATE:
        return "none"
    if psi < PSI_MAJOR:
        return "moderate"
    return "major"
