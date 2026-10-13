"""Deterministic paired uncertainty analysis."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Result of one paired bootstrap comparison."""

    sample_count: int
    bootstrap_samples: int
    seed: int

    baseline_mean: float
    selected_mean: float
    mean_difference: float

    confidence_level: float
    confidence_interval_low: float
    confidence_interval_high: float

    probability_difference_positive: float
    two_sided_p_value: float


def paired_bootstrap_difference(
    *,
    baseline_values: Sequence[float],
    selected_values: Sequence[float],
    bootstrap_samples: int = 10_000,
    seed: int = 20_261_003,
    confidence_level: float = 0.95,
) -> PairedBootstrapResult:
    """Estimate a paired mean difference by bootstrap."""

    if not baseline_values:
        raise ValueError("At least one paired observation is required.")

    if len(baseline_values) != len(selected_values):
        raise ValueError("Paired samples must have equal lengths.")

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")

    baseline = tuple(float(value) for value in baseline_values)
    selected = tuple(float(value) for value in selected_values)

    paired_differences = tuple(
        selected_value - baseline_value
        for baseline_value, selected_value in zip(
            baseline,
            selected,
            strict=True,
        )
    )

    sample_count = len(paired_differences)
    observed_difference = sum(paired_differences) / sample_count

    random_generator = random.Random(seed)
    bootstrap_differences: list[float] = []

    for _ in range(bootstrap_samples):
        bootstrap_difference = sum(
            paired_differences[random_generator.randrange(sample_count)]
            for _ in range(sample_count)
        ) / sample_count
        bootstrap_differences.append(bootstrap_difference)

    bootstrap_differences.sort()

    alpha = 1.0 - confidence_level

    low_index = int((alpha / 2.0) * bootstrap_samples)
    high_index = min(
        bootstrap_samples - 1,
        int((1.0 - alpha / 2.0) * bootstrap_samples),
    )

    non_positive_probability = sum(
        difference <= 0.0 for difference in bootstrap_differences
    ) / bootstrap_samples

    non_negative_probability = sum(
        difference >= 0.0 for difference in bootstrap_differences
    ) / bootstrap_samples

    two_sided_p_value = min(
        1.0,
        2.0 * min(non_positive_probability, non_negative_probability),
    )

    return PairedBootstrapResult(
        sample_count=sample_count,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        baseline_mean=sum(baseline) / sample_count,
        selected_mean=sum(selected) / sample_count,
        mean_difference=observed_difference,
        confidence_level=confidence_level,
        confidence_interval_low=bootstrap_differences[low_index],
        confidence_interval_high=bootstrap_differences[high_index],
        probability_difference_positive=sum(
            difference > 0.0 for difference in bootstrap_differences
        ) / bootstrap_samples,
        two_sided_p_value=two_sided_p_value,
    )
