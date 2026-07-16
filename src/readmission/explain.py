"""SHAP explanations for the tree-based model pipelines.

Turns a fitted pipeline into per-patient signed feature contributions (what pushed
this patient's risk up or down) and a global importance ranking. Uses TreeSHAP,
which is exact and fast for gradient-boosted trees.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _positive_class_shap(explainer: Any, matrix: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return SHAP values for the positive class as an (n_rows, n_features) array."""
    values = np.asarray(explainer.shap_values(matrix))
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        # (n_classes, n_rows, n_features) or (n_rows, n_features, n_classes)
        return values[1] if values.shape[0] == 2 else values[..., 1]
    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


def _shap_matrix(pipeline: Pipeline, frame: pd.DataFrame) -> tuple[npt.NDArray[Any], list[str]]:
    preprocess = pipeline[:-1]
    classifier = pipeline.named_steps["classifier"]
    matrix = preprocess.transform(frame)
    names = list(preprocess.get_feature_names_out())
    explainer = shap.TreeExplainer(classifier)
    return _positive_class_shap(explainer, matrix), names


def top_contributions(
    pipeline: Pipeline, frame: pd.DataFrame, *, top_k: int = 5
) -> list[list[dict[str, float | str]]]:
    """Per-row top-``k`` signed SHAP contributions (largest absolute effect first)."""
    shap_values, names = _shap_matrix(pipeline, frame)
    results: list[list[dict[str, float | str]]] = []
    for row in shap_values:
        order = np.argsort(np.abs(row))[::-1][:top_k]
        results.append(
            [
                {
                    "feature": names[j],
                    "shap": float(row[j]),
                    "direction": "increases" if row[j] > 0 else "decreases",
                }
                for j in order
            ]
        )
    return results


def global_importance(pipeline: Pipeline, frame: pd.DataFrame) -> dict[str, float]:
    """Mean absolute SHAP value per feature, ranked descending."""
    shap_values, names = _shap_matrix(pipeline, frame)
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(names, mean_abs, strict=True), key=lambda kv: kv[1], reverse=True)
    return {name: float(value) for name, value in ranked}
