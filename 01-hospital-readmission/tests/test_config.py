"""Tests for configuration loading."""

from __future__ import annotations

import pytest

from readmission.config import get_settings


def test_defaults() -> None:
    settings = get_settings()
    assert settings.random_seed == 42
    assert settings.first_encounter_only is True
    assert settings.raw_data_path.name == "diabetic_data.csv"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("READMISSION_RANDOM_SEED", "7")
    monkeypatch.setenv("READMISSION_FIRST_ENCOUNTER_ONLY", "false")
    settings = get_settings()
    assert settings.random_seed == 7
    assert settings.first_encounter_only is False
