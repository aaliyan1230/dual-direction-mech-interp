"""Unit tests for evaluation metrics helpers."""
from __future__ import annotations

from ddmi.evaluation.metrics import bootstrap_rate_ci


def test_bootstrap_rate_ci_reports_point_estimate_and_bounds() -> None:
    ci = bootstrap_rate_ci([True, True, False, False], num_bootstrap=200, seed=7)

    assert abs(ci["point_estimate"] - 0.5) < 1e-10
    assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
    assert ci["num_samples"] == 4
    assert ci["confidence"] == 0.95


def test_bootstrap_rate_ci_handles_empty_samples() -> None:
    ci = bootstrap_rate_ci([], num_bootstrap=50, seed=3)

    assert ci["point_estimate"] == 0.0
    assert ci["lower"] == 0.0
    assert ci["upper"] == 0.0
    assert ci["num_samples"] == 0