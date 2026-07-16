"""Tests for SHAP explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from readmission import explain
from readmission.models.boosted import build_xgboost_pipeline


class _FakeExplainer:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def shap_values(self, matrix: np.ndarray) -> np.ndarray:
        return self._values


def test_positive_class_shap_normalizes_3d_shapes() -> None:
    matrix = np.zeros((4, 5))
    # (n_classes, n_rows, n_features)
    by_class = explain._positive_class_shap(_FakeExplainer(np.ones((2, 4, 5))), matrix)
    assert by_class.shape == (4, 5)
    # (n_rows, n_features, n_classes)
    last_axis = explain._positive_class_shap(_FakeExplainer(np.ones((4, 5, 3))), matrix)
    assert last_axis.shape == (4, 5)


def test_positive_class_shap_rejects_unexpected_shape() -> None:
    with pytest.raises(ValueError, match="Unexpected SHAP"):
        explain._positive_class_shap(_FakeExplainer(np.ones((2, 2, 2, 2))), np.zeros((2, 2)))


def test_top_contributions_shape_and_fields(
    model_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    frame, y, _ = model_data
    pipe = build_xgboost_pipeline()
    pipe.fit(frame, y)
    contribs = explain.top_contributions(pipe, frame.head(3), top_k=4)
    assert len(contribs) == 3
    for row in contribs:
        assert len(row) <= 4
        assert all({"feature", "shap", "direction"} <= set(item) for item in row)


def test_global_importance_ranked_and_non_negative(
    model_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    frame, y, _ = model_data
    pipe = build_xgboost_pipeline()
    pipe.fit(frame, y)
    importance = explain.global_importance(pipe, frame)
    values = list(importance.values())
    assert len(values) > 0
    assert values == sorted(values, reverse=True)
    assert all(v >= 0 for v in values)
