"""Save and load fitted model pipelines with joblib."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, path: Path) -> Path:
    """Serialise a fitted model to ``path`` (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> Any:
    """Load a model saved by :func:`save_model`."""
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Train and save one first (python -m readmission.train)."
        )
    return joblib.load(path)
