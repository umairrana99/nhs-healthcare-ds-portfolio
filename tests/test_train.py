"""End-to-end test of the training entry point with monkeypatched data.

Exercises the full flow — ingestion cleaning -> feature engineering ->
preprocessing -> grouped CV -> baseline model -> metrics — without needing the
real (licensed) dataset on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from readmission import train


def _synthetic_raw() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 40
    return pd.DataFrame(
        {
            "encounter_id": np.arange(1, n + 1),
            "patient_nbr": np.arange(100, 100 + n),
            # 1-5 avoids the death/hospice dispositions that get dropped.
            "discharge_disposition_id": rng.integers(1, 6, n),
            "readmitted": rng.choice(["<30", ">30", "NO"], n, p=[0.4, 0.3, 0.3]),
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


def test_main_runs_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "load_raw", lambda _path: _synthetic_raw())
    metrics = train.main()
    assert {"auroc_mean", "auprc_mean", "n_splits"}.issubset(metrics)
    assert 0.0 <= metrics["auroc_mean"] <= 1.0
    assert 0.0 <= metrics["auprc_mean"] <= 1.0
