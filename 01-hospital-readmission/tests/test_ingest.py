"""Tests for data ingestion — with emphasis on the leakage-prevention rules."""

from __future__ import annotations

import pandas as pd

from readmission.constants import PATIENT_ID, TARGET_COLUMN
from readmission.data import ingest


def test_replace_missing_tokens_converts_question_mark(raw_df: pd.DataFrame) -> None:
    cleaned = ingest.replace_missing_tokens(raw_df)
    assert cleaned.loc[cleaned["patient_nbr"] == 500, "race"].isna().all()
    assert "?" not in cleaned["race"].dropna().to_numpy()


def test_drop_expired_hospice_removes_death_and_hospice(raw_df: pd.DataFrame) -> None:
    result = ingest.drop_expired_hospice(raw_df)
    # Rows with disposition 11 (expired) and 13 (hospice) must be gone.
    assert not result["discharge_disposition_id"].isin({11, 13, 14, 19, 20, 21}).any()
    # Patients 200 and 300 were the death/hospice rows.
    assert set(result["patient_nbr"]) == {100, 400, 500}


def test_binarize_target_only_marks_under_30_as_positive(raw_df: pd.DataFrame) -> None:
    result = ingest.binarize_target(raw_df)
    mapping = dict(zip(result["encounter_id"], result[TARGET_COLUMN], strict=True))
    assert mapping[1] == 1  # "<30"
    assert mapping[3] == 1  # "<30"
    assert mapping[2] == 0  # "NO"
    assert mapping[5] == 0  # ">30"  -> NOT positive for a 30-day model
    assert result[TARGET_COLUMN].dtype.kind in {"i", "u"}
    assert "readmitted" not in result.columns  # raw dropped by default


def test_keep_first_encounter_dedups_by_patient(raw_df: pd.DataFrame) -> None:
    result = ingest.keep_first_encounter_per_patient(raw_df)
    assert result[PATIENT_ID].is_unique
    # Patient 100's kept row is the earlier encounter (id 1), not id 2.
    kept = result.loc[result[PATIENT_ID] == 100, "encounter_id"].item()
    assert kept == 1


def test_build_dataset_end_to_end(raw_df: pd.DataFrame) -> None:
    result = ingest.build_dataset(raw_df, first_encounter_only=True)
    # No death/hospice rows, one row per patient, binary target present.
    assert result[PATIENT_ID].is_unique
    assert not result["discharge_disposition_id"].isin({11, 13}).any()
    assert set(result[TARGET_COLUMN].unique()) <= {0, 1}
    # 3 survivors after cleaning: patients 100 (<30 ->1), 400 (>30 ->0), 500 (>30 ->0)
    assert len(result) == 3
    assert result[TARGET_COLUMN].sum() == 1
