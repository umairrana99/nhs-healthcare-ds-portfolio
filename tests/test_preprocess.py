"""Tests for feature assembly and the preprocessing ColumnTransformer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmission import preprocess


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_in_hospital": [3, 7, 2, 5],
            "num_lab_procedures": [40.0, 55.0, np.nan, 30.0],  # NaN -> median impute
            "num_procedures": [1, 0, 2, 1],
            "num_medications": [10, 20, 8, 16],
            "number_outpatient": [0, 1, 0, 2],
            "number_emergency": [0, 0, 1, 0],
            "number_inpatient": [1, 0, 0, 3],
            "number_diagnoses": [9, 5, 7, 8],
            "race": ["Caucasian", None, "Asian", "Hispanic"],  # None -> Unknown
            "gender": ["Female", "Male", "Female", "Male"],
            "admission_type_id": [1, 2, 1, 3],
            "discharge_disposition_id": [1, 6, 1, 2],
            "admission_source_id": [7, 1, 7, 4],
            "insulin": ["Up", "No", "Steady", "Down"],
            "change": ["Ch", "No", "Ch", "No"],
            "diabetesMed": ["Yes", "Yes", "No", "Yes"],
            "medical_specialty": ["Cardiology", None, "InternalMedicine", None],
            "diag_1": ["410", "250.5", "486", "715"],
            "diag_2": ["250", "401", "V27", None],
            "diag_3": ["428", "250", "560", "E909"],
            "A1Cresult": ["None", ">8", "Norm", ">7"],
            "max_glu_serum": [">200", "None", "None", "Norm"],
            "age": ["[70-80)", "[50-60)", "[40-50)", "[80-90)"],
            "metformin": ["Steady", "No", "No", "Up"],
            "glipizide": ["No", "Down", "No", "No"],
        }
    )


def test_select_model_frame_fills_unknown_and_adds_features() -> None:
    out = preprocess.select_model_frame(_raw_frame())
    # Missing categoricals become the explicit "Unknown" category.
    assert out.loc[1, "race"] == "Unknown"
    assert out.loc[1, "medical_specialty"] == "Unknown"
    # Engineered + diagnosis-group columns exist.
    for col in ["total_prior_visits", "active_med_count", "age_midpoint", "diag_1_group"]:
        assert col in out.columns


def test_preprocessor_fit_transform_is_clean_numeric() -> None:
    frame = preprocess.select_model_frame(_raw_frame())
    pre = preprocess.build_preprocessor()
    matrix = pre.fit_transform(frame)
    assert matrix.shape[0] == 4  # rows preserved
    assert matrix.dtype.kind == "f"  # all numeric
    assert not np.isnan(matrix).any()  # NaN was imputed, nothing leaks through


def test_preprocessor_handles_unseen_categories() -> None:
    frame = preprocess.select_model_frame(_raw_frame())
    pre = preprocess.build_preprocessor()
    pre.fit(frame.iloc[:3])  # "Hispanic" race only appears in row 3
    transformed = pre.transform(frame.iloc[3:])
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()
