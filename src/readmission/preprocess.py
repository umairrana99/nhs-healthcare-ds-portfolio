"""Assemble model features and build the preprocessing transformer.

``select_model_frame`` turns a cleaned dataframe into the exact model input
(runs diagnosis grouping + feature engineering, then normalises categoricals with
an explicit "Unknown" for missing values). ``build_preprocessor`` returns an
unfitted scikit-learn ``ColumnTransformer`` to be fit inside cross-validation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from readmission.features.diagnoses import add_diagnosis_groups
from readmission.features.engineer import engineer_features
from readmission.features.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

_UNKNOWN = "Unknown"


def select_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model-input frame: diagnosis groups + engineered features.

    Categorical columns are cast to strings with missing values made an explicit
    ``"Unknown"`` category (their absence is informative on this dataset).
    """
    out = add_diagnosis_groups(df)
    out = engineer_features(out)
    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype("object")
            out[col] = out[col].where(out[col].notna(), _UNKNOWN).astype(str)
    return out


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the declared feature schema."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("num", numeric, list(NUMERIC_FEATURES)),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
            ("bin", "passthrough", list(BINARY_FEATURES)),
        ],
        remainder="drop",
    )
