"""FastAPI application exposing prediction and explanation endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI

from readmission import service
from readmission.api.dependencies import get_model
from readmission.api.schemas import (
    ExplanationResponse,
    FeatureContribution,
    PatientRecord,
    PredictionResponse,
)

app = FastAPI(title="Hospital Readmission Prediction API", version="0.1.0")

Model = Annotated[Any, Depends(get_model)]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: PatientRecord, model: Model) -> PredictionResponse:
    """Return the 30-day readmission probability and a risk band."""
    result = service.score(model, record)
    return PredictionResponse(
        readmission_probability=result["probability"], risk_band=result["risk_band"]
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain_prediction(record: PatientRecord, model: Model) -> ExplanationResponse:
    """Return the prediction plus the top SHAP feature contributions."""
    result = service.score(model, record)
    factors = [FeatureContribution(**contribution) for contribution in result["top_factors"]]
    return ExplanationResponse(readmission_probability=result["probability"], top_factors=factors)
