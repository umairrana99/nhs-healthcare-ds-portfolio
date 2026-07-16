"""Grouped cross-validation with imbalance-appropriate metrics.

Reports AUROC and AUPRC (discrimination) plus Brier score and ECE (calibration).
Accuracy is deliberately omitted — at ~11% prevalence a "never readmit" model is
~89% accurate and useless; AUPRC is the honest headline for the positive class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from readmission.cv import group_splits
from readmission.metrics import expected_calibration_error
from readmission.models.baseline import build_baseline_pipeline


def cross_validate(
    frame: pd.DataFrame,
    y: npt.NDArray[Any],
    groups: npt.NDArray[Any],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    pipeline_factory: Callable[[], Pipeline] = build_baseline_pipeline,
) -> dict[str, float]:
    """Fit the pipeline across grouped folds; return mean/std AUROC and AUPRC."""
    labels = np.asarray(y)
    aurocs: list[float] = []
    auprcs: list[float] = []
    briers: list[float] = []
    eces: list[float] = []
    for train_idx, test_idx in group_splits(
        labels, groups, n_splits=n_splits, random_state=random_state
    ):
        model = pipeline_factory()
        model.fit(frame.iloc[train_idx], labels[train_idx])
        proba = model.predict_proba(frame.iloc[test_idx])[:, 1]
        test_labels = labels[test_idx]
        aurocs.append(float(roc_auc_score(test_labels, proba)))
        auprcs.append(float(average_precision_score(test_labels, proba)))
        briers.append(float(brier_score_loss(test_labels, proba)))
        eces.append(expected_calibration_error(test_labels, proba))

    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "brier_mean": float(np.mean(briers)),
        "ece_mean": float(np.mean(eces)),
        "n_splits": float(n_splits),
    }


def compare_models(
    frame: pd.DataFrame,
    y: npt.NDArray[Any],
    groups: npt.NDArray[Any],
    factories: dict[str, Callable[[], Pipeline]],
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Cross-validate each named model factory; return its metrics keyed by name."""
    return {
        name: cross_validate(
            frame,
            y,
            groups,
            n_splits=n_splits,
            random_state=random_state,
            pipeline_factory=factory,
        )
        for name, factory in factories.items()
    }
