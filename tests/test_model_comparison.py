"""Tests for the extended evaluator (calibration metrics + model comparison)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmission import evaluate
from readmission.models.baseline import build_baseline_pipeline
from readmission.models.boosted import build_lightgbm_pipeline


def test_cross_validate_reports_calibration_metrics(
    model_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    frame, y, groups = model_data
    metrics = evaluate.cross_validate(frame, y, groups, n_splits=3)
    for key in ["auroc_mean", "auprc_mean", "brier_mean", "ece_mean"]:
        assert key in metrics
    assert 0.0 <= metrics["brier_mean"] <= 1.0
    assert 0.0 <= metrics["ece_mean"] <= 1.0


def test_compare_models_returns_metrics_per_model(
    model_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    frame, y, groups = model_data
    results = evaluate.compare_models(
        frame,
        y,
        groups,
        {"logreg": build_baseline_pipeline, "lightgbm": build_lightgbm_pipeline},
        n_splits=3,
    )
    assert set(results) == {"logreg", "lightgbm"}
    for model_metrics in results.values():
        assert 0.0 <= model_metrics["auroc_mean"] <= 1.0
