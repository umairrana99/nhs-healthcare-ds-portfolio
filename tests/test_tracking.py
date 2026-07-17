"""Tests for optional MLflow tracking (skipped when mlflow is not installed)."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_log_run_records_params_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    from readmission import tracking

    run_id = tracking.log_run(
        {"model": "xgboost", "n_estimators": "300"},
        {"auroc": 0.82, "auprc": 0.41},
        experiment="test",
        tracking_uri=(tmp_path / "mlruns").as_uri(),
    )
    assert run_id
    run = mlflow.get_run(run_id)
    assert run.data.metrics["auroc"] == 0.82
    assert run.data.params["model"] == "xgboost"
