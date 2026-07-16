"""Gradient-boosted model pipelines (XGBoost and LightGBM).

Both wrap the shared preprocessor and handle the ~11% positive rate: LightGBM via
``class_weight="balanced"``, XGBoost via ``scale_pos_weight`` (defaulted to the
dataset's roughly 8:1 negative:positive ratio).
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from readmission.preprocess import build_preprocessor


def build_xgboost_pipeline(*, scale_pos_weight: float = 8.0, random_state: int = 42) -> Pipeline:
    """Preprocessing + XGBoost classifier."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="aucpr",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_lightgbm_pipeline(*, random_state: int = 42) -> Pipeline:
    """Preprocessing + LightGBM classifier."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=300,
                    num_leaves=31,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )
