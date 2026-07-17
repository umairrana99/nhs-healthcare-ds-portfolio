"""Request and response models for the API."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

DISCLAIMER = "Decision support only; not a substitute for clinical judgement."


class PatientRecord(BaseModel):
    """One patient's features at discharge (raw, pre-engineering)."""

    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    gender: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    diag_1: str
    age: str
    race: str | None = None
    medical_specialty: str | None = None
    diag_2: str | None = None
    diag_3: str | None = None
    insulin: str = "No"
    metformin: str = "No"
    glipizide: str = "No"
    change: str = "No"
    diabetesMed: str = "No"
    A1Cresult: str = "None"
    max_glu_serum: str = "None"

    def to_frame(self) -> pd.DataFrame:
        """Represent the record as a single-row DataFrame for the pipeline."""
        return pd.DataFrame([self.model_dump()])


class FeatureContribution(BaseModel):
    feature: str
    shap: float
    direction: str


class PredictionResponse(BaseModel):
    readmission_probability: float = Field(ge=0.0, le=1.0)
    risk_band: str
    disclaimer: str = DISCLAIMER


class ExplanationResponse(BaseModel):
    readmission_probability: float = Field(ge=0.0, le=1.0)
    top_factors: list[FeatureContribution]
    disclaimer: str = DISCLAIMER
