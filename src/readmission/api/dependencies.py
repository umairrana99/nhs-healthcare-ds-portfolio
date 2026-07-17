"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Any

from readmission.config import get_settings
from readmission.persist import load_model


def get_model() -> Any:
    """Load the served model from the configured path.

    Overridden in tests via ``app.dependency_overrides``.
    """
    return load_model(get_settings().model_path)
