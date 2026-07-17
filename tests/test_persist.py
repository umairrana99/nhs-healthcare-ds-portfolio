"""Tests for model persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from readmission import persist


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    x = np.array([[0.0], [1.0], [0.0], [1.0]])
    model = LogisticRegression().fit(x, [0, 1, 0, 1])
    path = persist.save_model(model, tmp_path / "sub" / "model.joblib")
    assert path.exists()

    loaded = persist.load_model(path)
    assert loaded.predict([[1.0]])[0] == model.predict([[1.0]])[0]


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        persist.load_model(tmp_path / "missing.joblib")
