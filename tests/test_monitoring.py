"""Tests for drift monitoring."""

from __future__ import annotations

import numpy as np

from readmission.monitoring import drift_label, population_stability_index


def test_psi_near_zero_for_identical_distribution() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(size=2000)
    assert population_stability_index(reference, reference.copy()) < 0.1


def test_psi_large_for_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(0.0, 1.0, 2000)
    shifted = rng.normal(3.0, 1.0, 2000)
    assert population_stability_index(reference, shifted) > 0.25


def test_psi_constant_reference_is_zero() -> None:
    constant = np.ones(100)
    assert population_stability_index(constant, np.ones(100)) == 0.0


def test_drift_label_thresholds() -> None:
    assert drift_label(0.05) == "none"
    assert drift_label(0.15) == "moderate"
    assert drift_label(0.40) == "major"
