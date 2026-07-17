"""Shared scoring logic used by both the API and the dashboard."""

from __future__ import annotations

from typing import Any

from readmission import explain
from readmission.api.schemas import PatientRecord
from readmission.preprocess import select_model_frame


def risk_band(probability: float) -> str:
    """Bucket a probability into Low / Moderate / High."""
    if probability < 0.10:
        return "Low"
    if probability < 0.30:
        return "Moderate"
    return "High"


def score(model: Any, record: PatientRecord) -> dict[str, Any]:
    """Return probability, risk band, and top SHAP contributions for one patient."""
    frame = select_model_frame(record.to_frame())
    probability = float(model.predict_proba(frame)[0, 1])
    return {
        "probability": probability,
        "risk_band": risk_band(probability),
        "top_factors": explain.top_contributions(model, frame, top_k=5)[0],
    }
