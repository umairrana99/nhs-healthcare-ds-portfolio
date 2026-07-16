"""Shared test fixtures.

The fixture mirrors the real schema on a handful of rows so the cleaning logic
can be tested without downloading the licensed dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from readmission import preprocess


@pytest.fixture
def model_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A model-ready (frame, y, groups) sample for training/evaluation tests.

    30 patients, 3 encounters each; label is constant per patient.
    """
    rng = np.random.default_rng(0)
    n_patients, per_patient = 30, 3
    n = n_patients * per_patient
    groups = np.repeat(np.arange(n_patients), per_patient)
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


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A tiny raw-shaped frame covering the cases the cleaning logic must handle.

    Rows:
      1 (patient 100) survives, readmitted "<30"  -> positive, kept
      2 (patient 100) later encounter, "NO"       -> dropped by first-encounter filter
      3 (patient 200) expired (disposition 11)     -> dropped (death)
      4 (patient 300) hospice (disposition 13)     -> dropped (hospice)
      5 (patient 400) readmitted ">30"             -> negative, kept
      6 (patient 500) has "?" race + ">30"         -> "?" -> NA, negative, kept
    """
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4, 5, 6],
            "patient_nbr": [100, 100, 200, 300, 400, 500],
            "race": ["Caucasian", "Caucasian", "AfricanAmerican", "Caucasian", "Asian", "?"],
            "discharge_disposition_id": [1, 1, 11, 13, 1, 1],
            "readmitted": ["<30", "NO", "<30", "NO", ">30", ">30"],
        }
    )
