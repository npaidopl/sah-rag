"""Tests for paired retrieval uncertainty analysis."""

from __future__ import annotations

import pytest

from rag_system.evaluation import (
    paired_bootstrap_difference,
)


def test_paired_bootstrap_is_deterministic() -> None:
    """Equivalent inputs must produce equal results."""

    baseline = (0.0, 0.2, 0.4, 0.6)
    selected = (0.1, 0.3, 0.5, 0.7)

    first = paired_bootstrap_difference(
        baseline_values=baseline,
        selected_values=selected,
        bootstrap_samples=1_000,
        seed=123,
    )

    second = paired_bootstrap_difference(
        baseline_values=baseline,
        selected_values=selected,
        bootstrap_samples=1_000,
        seed=123,
    )

    assert first == second
    assert first.mean_difference == pytest.approx(0.1)


def test_positive_difference_has_positive_interval() -> None:
    """A consistent improvement should exclude zero."""

    result = paired_bootstrap_difference(
        baseline_values=(0.0, 0.1, 0.2, 0.3),
        selected_values=(0.2, 0.3, 0.4, 0.5),
        bootstrap_samples=2_000,
        seed=456,
    )

    assert result.confidence_interval_low > 0.0
    assert result.probability_difference_positive == 1.0


def test_paired_bootstrap_rejects_misaligned_samples() -> None:
    """Paired samples must have equal lengths."""

    with pytest.raises(
        ValueError,
        match="equal lengths",
    ):
        paired_bootstrap_difference(
            baseline_values=(0.1,),
            selected_values=(0.1, 0.2),
        )
