"""Tests for calibration metrics."""

from __future__ import annotations

import numpy as np

from readmission.metrics import expected_calibration_error


def test_ece_zero_for_perfectly_calibrated() -> None:
    y_prob = np.array([0.0] * 50 + [1.0] * 50)
    y_true = np.array([0] * 50 + [1] * 50)
    assert expected_calibration_error(y_true, y_prob) == 0.0


def test_ece_large_for_confidently_wrong() -> None:
    y_prob = np.full(100, 0.9)
    y_true = np.zeros(100, dtype=int)
    assert expected_calibration_error(y_true, y_prob) > 0.8


def test_ece_within_unit_range() -> None:
    rng = np.random.default_rng(0)
    ece = expected_calibration_error(rng.integers(0, 2, 200), rng.random(200))
    assert 0.0 <= ece <= 1.0
