"""Tests for engineered clinical features."""

from __future__ import annotations

import pandas as pd

from readmission.features import engineer


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "number_outpatient": [0, 2],
            "number_emergency": [1, 0],
            "number_inpatient": [0, 3],
            "num_medications": [8, 20],
            "metformin": ["Steady", "No"],
            "insulin": ["Up", "No"],
            "glipizide": ["No", "Down"],
            "A1Cresult": ["None", ">8"],
            "max_glu_serum": [">200", "None"],
            "age": ["[40-50)", "[70-80)"],
        }
    )


def test_prior_utilisation() -> None:
    out = engineer.add_prior_utilisation(_sample())
    assert out["total_prior_visits"].tolist() == [1, 5]
    assert out["any_prior_inpatient"].tolist() == [0, 1]


def test_medication_features() -> None:
    out = engineer.add_medication_features(_sample())
    # Row 0: metformin=Steady, insulin=Up -> 2 active. Row 1: glipizide=Down -> 1.
    assert out["active_med_count"].tolist() == [2, 1]
    assert out["polypharmacy"].tolist() == [0, 1]


def test_glycemic_features() -> None:
    out = engineer.add_glycemic_features(_sample())
    assert out["a1c_measured"].tolist() == [0, 1]
    assert out["a1c_level"].tolist() == [0, 3]  # None->0, >8->3
    assert out["glucose_measured"].tolist() == [1, 0]
    assert out["glucose_level"].tolist() == [2, 0]  # >200->2, None->0


def test_age_midpoint() -> None:
    out = engineer.add_age_midpoint(_sample())
    assert out["age_midpoint"].tolist() == [45.0, 75.0]


def test_engineer_features_is_non_mutating_and_complete() -> None:
    df = _sample()
    out = engineer.engineer_features(df)
    for col in ["total_prior_visits", "active_med_count", "a1c_level", "age_midpoint"]:
        assert col in out.columns
    # Original untouched.
    assert "total_prior_visits" not in df.columns


def test_age_midpoint_handles_missing_and_malformed() -> None:
    out = engineer.add_age_midpoint(pd.DataFrame({"age": ["[70-80)", None, "unknown"]}))
    vals = out["age_midpoint"].tolist()
    assert vals[0] == 75.0
    assert pd.isna(vals[1]) and pd.isna(vals[2])
