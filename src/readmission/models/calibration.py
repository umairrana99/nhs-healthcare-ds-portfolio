"""Probability calibration for the model pipelines.

Boosted trees are typically over-confident, and the output must be a trustworthy
probability because it drives resource-allocation decisions. This wraps a base
pipeline in ``CalibratedClassifierCV``, which fits the calibrator on held-out
folds of the training data only.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


def build_calibrated(
    base_factory: Callable[[], Pipeline],
    *,
    method: str = "isotonic",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Wrap a fresh base pipeline in probability calibration."""
    return CalibratedClassifierCV(base_factory(), method=method, cv=cv)
