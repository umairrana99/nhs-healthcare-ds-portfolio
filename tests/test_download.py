"""Tests for the dataset download helper (pure, non-network parts)."""

from __future__ import annotations

from pathlib import Path

from readmission.data import download


def test_get_download_instructions_mentions_url_and_file() -> None:
    text = download.get_download_instructions(Path("data") / "raw")
    assert download.DATASET_URL in text
    assert "diabetic_data.csv" in text
