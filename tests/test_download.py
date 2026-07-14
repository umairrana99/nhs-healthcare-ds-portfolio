"""Tests for the dataset download helper (network mocked, no real requests)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from readmission.data import download


def test_get_download_instructions_mentions_url_and_file() -> None:
    text = download.get_download_instructions(Path("data") / "raw")
    assert download.DATASET_URL in text
    assert "diabetic_data.csv" in text


def _fake_archive() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(download.CSV_MEMBER, "encounter_id,readmitted\n1,NO\n")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_download_dataset_extracts_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda _url: _FakeResponse(_fake_archive()))
    target = download.download_dataset(tmp_path)
    assert target == tmp_path / "diabetic_data.csv"
    assert target.read_text().startswith("encounter_id")


def test_download_dataset_skips_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    existing = tmp_path / "diabetic_data.csv"
    existing.write_text("already here")

    def _boom(_url: object) -> object:
        raise AssertionError("network must not be used when the file already exists")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    target = download.download_dataset(tmp_path)
    assert target == existing
    assert target.read_text() == "already here"
