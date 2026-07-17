"""Tests for the shared scoring service."""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmission import service
from readmission.api.schemas import PatientRecord
from readmission.models.boosted import build_xgboost_pipeline


def test_risk_band_thresholds() -> None:
    assert service.risk_band(0.05) == "Low"
    assert service.risk_band(0.20) == "Moderate"
    assert service.risk_band(0.50) == "High"


def _record() -> PatientRecord:
    return PatientRecord(
        time_in_hospital=5,
        num_lab_procedures=40,
        num_procedures=1,
        num_medications=16,
        number_outpatient=0,
        number_emergency=1,
        number_inpatient=2,
        number_diagnoses=9,
        gender="Female",
        admission_type_id=1,
        discharge_disposition_id=1,
        admission_source_id=7,
        diag_1="410",
        age="[70-80)",
    )


def test_score_returns_probability_band_and_factors(
    model_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    frame, y, _ = model_data
    model = build_xgboost_pipeline().fit(frame, y)
    result = service.score(model, _record())
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_band"] in {"Low", "Moderate", "High"}
    assert len(result["top_factors"]) >= 1
