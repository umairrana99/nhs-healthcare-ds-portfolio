"""Configuration via Pydantic Settings.

All settings are overridable through environment variables prefixed with
``READMISSION_`` or a local ``.env`` file (never committed).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the readmission pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="READMISSION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Paths ---
    project_root: Path = _PROJECT_ROOT
    raw_data_path: Path = _PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
    processed_data_dir: Path = _PROJECT_ROOT / "data" / "processed"
    model_path: Path = _PROJECT_ROOT / "models" / "model.joblib"

    # --- Reproducibility ---
    random_seed: int = 42

    # --- Data preparation ---
    # Keep only the first encounter per patient so rows are statistically
    # independent (Strack et al.). Set False to model all encounters with a
    # patient-grouped split instead.
    first_encounter_only: bool = True

    # --- Logging ---
    log_level: str = Field(default="INFO")


def get_settings() -> Settings:
    """Return a fresh Settings instance (reads env / .env at call time)."""
    return Settings()
