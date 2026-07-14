"""End-to-end smoke test: features -> preprocessing -> grouped CV -> baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmission import evaluate, preprocess
from readmission.models.baseline import build_baseline_pipeline


def _synthetic() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n_patients, per_patient = 30, 3
    n = n_patients * per_patient
    groups = np.repeat(np.arange(n_patients), per_patient)
    # Consistent label per patient (a patient is readmitted or not).
    y = np.repeat(rng.integers(0, 2, n_patients), per_patient)
    raw = pd.DataFrame(
        {
            "time_in_hospital": rng.integers(1, 14, n),
            "num_lab_procedures": rng.integers(1, 80, n).astype(float),
            "num_procedures": rng.integers(0, 6, n),
            "num_medications": rng.integers(1, 40, n),
            "number_outpatient": rng.integers(0, 5, n),
            "number_emergency": rng.integers(0, 3, n),
            "number_inpatient": rng.integers(0, 4, n),
            "number_diagnoses": rng.integers(1, 16, n),
            "race": rng.choice(["Caucasian", "AfricanAmerican", "Asian", None], n),
            "gender": rng.choice(["Male", "Female"], n),
            "admission_type_id": rng.integers(1, 6, n),
            "discharge_disposition_id": rng.integers(1, 10, n),
            "admission_source_id": rng.integers(1, 10, n),
            "insulin": rng.choice(["No", "Steady", "Up", "Down"], n),
            "change": rng.choice(["No", "Ch"], n),
            "diabetesMed": rng.choice(["No", "Yes"], n),
            "medical_specialty": rng.choice(["Cardiology", "InternalMedicine", None], n),
            "diag_1": rng.choice(["410", "250.5", "486", "715", "200"], n),
            "diag_2": rng.choice(["250", "401", "V27", "560"], n),
            "diag_3": rng.choice(["428", "250", "560", "E909"], n),
            "A1Cresult": rng.choice(["None", "Norm", ">7", ">8"], n),
            "max_glu_serum": rng.choice(["None", "Norm", ">200", ">300"], n),
            "age": rng.choice(["[40-50)", "[50-60)", "[70-80)", "[80-90)"], n),
            "metformin": rng.choice(["No", "Steady", "Up"], n),
            "glipizide": rng.choice(["No", "Down"], n),
        }
    )
    return preprocess.select_model_frame(raw), y, groups


def test_baseline_pipeline_fits_and_predicts_proba() -> None:
    frame, y, _ = _synthetic()
    pipe = build_baseline_pipeline()
    pipe.fit(frame, y)
    proba = pipe.predict_proba(frame)[:, 1]
    assert proba.shape == (len(y),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_cross_validate_returns_valid_metrics() -> None:
    frame, y, groups = _synthetic()
    metrics = evaluate.cross_validate(frame, y, groups, n_splits=3)
    for key in ["auroc_mean", "auroc_std", "auprc_mean", "auprc_std"]:
        assert key in metrics
    assert 0.0 <= metrics["auroc_mean"] <= 1.0
    assert 0.0 <= metrics["auprc_mean"] <= 1.0
    assert metrics["n_splits"] == 3.0
