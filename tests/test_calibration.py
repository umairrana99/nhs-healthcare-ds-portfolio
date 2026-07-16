"""Tests for probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from readmission.models.baseline import build_baseline_pipeline
from readmission.models.calibration import build_calibrated


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrated_model_predicts_valid_proba(
    method: str, model_data: tuple[pd.DataFrame, np.ndarray, np.ndarray]
) -> None:
    frame, y, _ = model_data
    model = build_calibrated(build_baseline_pipeline, method=method, cv=3)
    model.fit(frame, y)
    proba = model.predict_proba(frame)[:, 1]
    assert proba.shape == (len(y),)
    assert ((proba >= 0) & (proba <= 1)).all()
