"""Unit tests for linear probe helpers."""
from __future__ import annotations

from ddmi.evaluation.probes import fit_and_evaluate_binary_probe


def test_fit_and_evaluate_binary_probe_separates_easy_data() -> None:
    positive = [[2.0, 2.0], [2.5, 1.5], [1.8, 2.2], [2.2, 1.9]]
    negative = [[-2.0, -2.0], [-1.5, -2.5], [-2.2, -1.8], [-1.9, -2.1]]

    result = fit_and_evaluate_binary_probe(
        positive,
        negative,
        train_fraction=0.75,
        seed=11,
        epochs=250,
    )

    assert result["train"]["accuracy"] >= 0.99
    assert result["test"]["accuracy"] >= 0.99
    assert result["num_train"] == 6
    assert result["num_test"] == 2