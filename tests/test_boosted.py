"""Tests for the gradient-boosted model pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from readmission.models import boosted


@pytest.mark.parametrize(
    "factory",
    [boosted.build_xgboost_pipeline, boosted.build_lightgbm_pipeline],
)
def test_boosted_pipeline_fits_and_predicts_proba(
    factory: object, model_data: tuple[pd.DataFrame, np.ndarray, np.ndarray]
) -> None:
    frame, y, _ = model_data
    pipe = factory()  # type: ignore[operator]
    pipe.fit(frame, y)
    proba = pipe.predict_proba(frame)[:, 1]
    assert proba.shape == (len(y),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_xgboost_accepts_scale_pos_weight(
    model_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    frame, y, _ = model_data
    pipe = boosted.build_xgboost_pipeline(scale_pos_weight=3.0)
    pipe.fit(frame, y)
    assert pipe.named_steps["classifier"].get_params()["scale_pos_weight"] == 3.0
