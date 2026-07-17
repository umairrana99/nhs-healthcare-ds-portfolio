"""FastAPI application exposing prediction and explanation endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI

from readmission import explain
from readmission.api.dependencies import get_model
from readmission.api.schemas import (
    ExplanationResponse,
    FeatureContribution,
    PatientRecord,
    PredictionResponse,
)
from readmission.preprocess import select_model_frame

app = FastAPI(title="Hospital Readmission Prediction API", version="0.1.0")

Model = Annotated[Any, Depends(get_model)]


def _risk_band(probability: float) -> str:
    if probability < 0.10:
        return "Low"
    if probability < 0.30:
        return "Moderate"
    return "High"


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: PatientRecord, model: Model) -> PredictionResponse:
    """Return the calibrated 30-day readmission probability and a risk band."""
    frame = select_model_frame(record.to_frame())
    probability = float(model.predict_proba(frame)[0, 1])
    return PredictionResponse(
        readmission_probability=probability, risk_band=_risk_band(probability)
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain_prediction(record: PatientRecord, model: Model) -> ExplanationResponse:
    """Return the prediction plus the top SHAP feature contributions."""
    frame = select_model_frame(record.to_frame())
    probability = float(model.predict_proba(frame)[0, 1])
    contributions = explain.top_contributions(model, frame, top_k=5)[0]
    factors = [FeatureContribution(**contribution) for contribution in contributions]
    return ExplanationResponse(readmission_probability=probability, top_factors=factors)
