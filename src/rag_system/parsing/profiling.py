"""Profiling utilities for parsed canonical documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median

from rag_system.artifacts.models import ParsedDocument


@dataclass(slots=True)
class ParserProfileAccumulator:
    """Incrementally aggregate deterministic parser statistics."""

    dataset_id: str

    document_count: int = 0
    parent_count: int = 0
    empty_document_count: int = 0
    single_parent_document_count: int = 0
    heading_parent_count: int = 0
    very_short_document_count: int = 0

    document_lengths: list[int] = field(
        default_factory=list
    )
    parent_lengths: list[int] = field(
        default_factory=list
    )
    parents_per_document: list[int] = field(
        default_factory=list
    )

    very_short_threshold: int = 50

    def add(
        self,
        parsed_document: ParsedDocument,
    ) -> None:
        """Add one parsed document to the profile."""

        document_length = (
            parsed_document
            .canonical_text_normalized_length
        )
        number_of_parents = len(
            parsed_document.parents
        )

        self.document_count += 1
        self.parent_count += number_of_parents

        self.document_lengths.append(document_length)
        self.parents_per_document.append(
            number_of_parents
        )

        if number_of_parents == 0:
            self.empty_document_count += 1

        if number_of_parents == 1:
            self.single_parent_document_count += 1

        if document_length < self.very_short_threshold:
            self.very_short_document_count += 1

        for parent in parsed_document.parents:
            self.parent_lengths.append(
                parent.normalized_char_span.length
            )

            if parent.metadata.get(
                "heading_detected"
            ) is True:
                self.heading_parent_count += 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible summary statistics."""

        return {
            "dataset_id": self.dataset_id,
            "document_count": self.document_count,
            "parent_count": self.parent_count,
            "empty_document_count": (
                self.empty_document_count
            ),
            "single_parent_document_count": (
                self.single_parent_document_count
            ),
            "heading_parent_count": (
                self.heading_parent_count
            ),
            "very_short_document_count": (
                self.very_short_document_count
            ),
            "average_parents_per_document": (
                _safe_mean(
                    self.parents_per_document
                )
            ),
            "median_parents_per_document": (
                _safe_median(
                    self.parents_per_document
                )
            ),
            "minimum_document_characters": (
                _safe_min(self.document_lengths)
            ),
            "median_document_characters": (
                _safe_median(
                    self.document_lengths
                )
            ),
            "maximum_document_characters": (
                _safe_max(self.document_lengths)
            ),
            "minimum_parent_characters": (
                _safe_min(self.parent_lengths)
            ),
            "median_parent_characters": (
                _safe_median(self.parent_lengths)
            ),
            "maximum_parent_characters": (
                _safe_max(self.parent_lengths)
            ),
        }


def _safe_mean(values: list[int]) -> float:
    """Return a mean or zero for an empty collection."""

    if not values:
        return 0.0

    return round(mean(values), 3)


def _safe_median(values: list[int]) -> float:
    """Return a median or zero for an empty collection."""

    if not values:
        return 0.0

    return float(median(values))


def _safe_min(values: list[int]) -> int:
    """Return a minimum or zero for an empty collection."""

    return min(values, default=0)


def _safe_max(values: list[int]) -> int:
    """Return a maximum or zero for an empty collection."""

    return max(values, default=0)