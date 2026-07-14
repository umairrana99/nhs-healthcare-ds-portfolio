"""Tests for ICD-9 diagnosis grouping."""

from __future__ import annotations

import pandas as pd
import pytest

from readmission.features import diagnoses


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("250.83", "Diabetes"),
        ("250", "Diabetes"),
        ("410", "Circulatory"),
        ("785", "Circulatory"),
        ("486", "Respiratory"),
        ("560", "Digestive"),
        ("999", "Injury"),
        ("715", "Musculoskeletal"),
        ("600", "Genitourinary"),
        ("788", "Genitourinary"),
        ("200", "Neoplasms"),
        ("V27", "Other"),
        ("E909", "Other"),
        ("789", "Other"),  # symptom code outside mapped ranges
    ],
)
def test_group_icd9_known_codes(code: str, expected: str) -> None:
    assert diagnoses.group_icd9(code) == expected


def test_group_icd9_missing() -> None:
    assert diagnoses.group_icd9(pd.NA) == "Missing"
    assert diagnoses.group_icd9(None) == "Missing"


def test_add_diagnosis_groups_adds_columns() -> None:
    df = pd.DataFrame({"diag_1": ["410", "250.5"], "diag_2": ["V27", pd.NA]})
    out = diagnoses.add_diagnosis_groups(df)
    assert out["diag_1_group"].tolist() == ["Circulatory", "Diabetes"]
    assert out["diag_2_group"].tolist() == ["Other", "Missing"]
    # Original frame is untouched.
    assert "diag_1_group" not in df.columns


def test_all_outputs_are_declared_groups() -> None:
    for code in ["410", "250", "V1", "abc", pd.NA, "999"]:
        assert diagnoses.group_icd9(code) in diagnoses.DIAGNOSIS_GROUPS
