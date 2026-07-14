"""Declared feature groups for the preprocessing pipeline.

Keeping the column lists in one place lets the preprocessor, the model, and the
tests agree on exactly which engineered/raw columns feed the model and how each
should be treated.
"""

from __future__ import annotations

# Continuous / ordinal columns -> median-impute + scale.
NUMERIC_FEATURES: tuple[str, ...] = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "total_prior_visits",
    "active_med_count",
    "age_midpoint",
    "a1c_level",
    "glucose_level",
)

# Already 0/1 -> passed through untouched.
BINARY_FEATURES: tuple[str, ...] = (
    "any_prior_inpatient",
    "polypharmacy",
    "a1c_measured",
    "glucose_measured",
)

# Discrete columns -> "Unknown"-filled and one-hot encoded.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "insulin",
    "change",
    "diabetesMed",
    "medical_specialty",
)

ALL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
