"""Baseline model: preprocessing + class-weighted Logistic Regression.

Logistic Regression is the interpretable reference every later model must beat.
``class_weight="balanced"`` handles the ~11% positive rate without resampling
(which would distort calibration on this tabular data).
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from readmission.preprocess import build_preprocessor


def build_baseline_pipeline(*, random_state: int = 42) -> Pipeline:
    """Return an unfitted preprocessing + Logistic Regression pipeline."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
