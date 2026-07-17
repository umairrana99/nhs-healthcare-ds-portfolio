"""Tests for the FastAPI service (model injected via dependency override)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from readmission import persist
from readmission.api import dependencies
from readmission.api.app import _risk_band, app
from readmission.models.boosted import build_xgboost_pipeline


@pytest.fixture
def client(model_data: tuple[pd.DataFrame, np.ndarray, np.ndarray]) -> Iterator[TestClient]:
    frame, y, _ = model_data
    model = build_xgboost_pipeline().fit(frame, y)
    app.dependency_overrides[dependencies.get_model] = lambda: model
    yield TestClient(app)
    app.dependency_overrides.clear()


def _payload() -> dict[str, object]:
    return {
        "time_in_hospital": 5,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 16,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
        "number_diagnoses": 9,
        "gender": "Female",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "diag_1": "410",
        "age": "[70-80)",
        "race": "Caucasian",
        "medical_specialty": "Cardiology",
        "diag_2": "250",
        "diag_3": "428",
        "insulin": "Up",
        "metformin": "Steady",
        "glipizide": "No",
        "change": "Ch",
        "diabetesMed": "Yes",
        "A1Cresult": ">8",
        "max_glu_serum": "None",
    }


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability_and_band(client: TestClient) -> None:
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["readmission_probability"] <= 1.0
    assert body["risk_band"] in {"Low", "Moderate", "High"}


def test_predict_rejects_incomplete_record(client: TestClient) -> None:
    response = client.post("/predict", json={"gender": "Female"})
    assert response.status_code == 422


def test_explain_returns_top_factors(client: TestClient) -> None:
    response = client.post("/explain", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["readmission_probability"] <= 1.0
    assert len(body["top_factors"]) >= 1
    assert {"feature", "shap", "direction"} <= set(body["top_factors"][0])


def test_risk_band_thresholds() -> None:
    assert _risk_band(0.05) == "Low"
    assert _risk_band(0.20) == "Moderate"
    assert _risk_band(0.50) == "High"


def test_get_model_loads_from_configured_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = LogisticRegression().fit(np.array([[0.0], [1.0]]), [0, 1])
    path = persist.save_model(model, tmp_path / "model.joblib")
    monkeypatch.setenv("READMISSION_MODEL_PATH", str(path))
    loaded = dependencies.get_model()
    assert loaded.predict([[1.0]])[0] == 1
