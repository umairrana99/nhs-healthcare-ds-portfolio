"""Optional MLflow experiment tracking.

MLflow is an optional dependency — install with ``pip install '.[tracking]'``. It is
imported lazily so the core package, API, and CI stay lean.
"""

from __future__ import annotations

from typing import Any


def log_run(
    params: dict[str, Any],
    metrics: dict[str, float],
    *,
    experiment: str = "readmission",
    tracking_uri: str | None = None,
    run_name: str | None = None,
) -> str:
    """Log parameters and metrics to MLflow; return the run id."""
    import mlflow  # optional dependency, imported lazily

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        return str(run.info.run_id)
