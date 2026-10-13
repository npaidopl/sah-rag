"""Deterministic LongRAG adapter models and long-unit construction.

This module implements project-owned adapter infrastructure.

It does not import, execute, or modify the audited LongRAG source
repository. Outputs created by this module are interface-validation
artifacts and are not paper-faithful LongRAG results.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import Field, model_validator

from rag_system.artifacts import (
    ArtifactBaseModel,
    normalize_identifier,
    short_fingerprint,
)

LongUnitTokenCounter = Callable[[str], int]

LongUnitConstructionPolicy = Literal[
    "adjacent_segments",
    "adjacent_document_segments",
    "single_document_if_within_budget",
    "structure_preserving_parent_groups",
]


class LongRAGSourceSegment(ArtifactBaseModel):
    """One source segment supplied to long-unit construction."""

    dataset_id: str

    doc_id: str
    parent_id: str
    chunk_id: str

    document_order: int = Field(ge=0)
    parent_order: int = Field(ge=0)
    chunk_order: int = Field(ge=0)

    text: str
    token_count: int = Field(gt=0)

    metadata: dict[str, object] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_source_segment(
        self,
    ) -> LongRAGSourceSegment:
        """Require complete source identity and text."""
        required_values = (
            self.dataset_id,
            self.doc_id,
            self.parent_id,
            self.chunk_id,
            self.text,
        )
        if any(
            not value.strip()
            for value in required_values
        ):
            raise ValueError(
                "LongRAG source identity and text "
                "cannot be empty."
            )
        return self


class LongUnit(ArtifactBaseModel):
    """One constructed long retrieval unit."""

    long_unit_id: str
    dataset_id: str
    construction_policy: LongUnitConstructionPolicy
    unit_order: int = Field(ge=0)

    text: str
    token_count: int = Field(gt=0)

    member_doc_ids: tuple[str, ...]
    member_parent_ids: tuple[str, ...]
    member_chunk_ids: tuple[str, ...]

    metadata: dict[str, object] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_long_unit(self) -> LongUnit:
        """Require non-empty memberships and consistent counts."""
        if not self.member_doc_ids:
            raise ValueError(
                "Long unit must contain at least "
                "one member document."
            )
        if not self.member_parent_ids:
            raise ValueError(
                "Long unit must contain at least "
                "one member parent."
            )
        if not self.member_chunk_ids:
            raise ValueError(
                "Long unit must contain at least "
                "one member chunk."
            )
        return self


class RankedLongUnit(ArtifactBaseModel):
    """One retrieved long unit with similarity score and rank."""

    query_id: str
    rank: int = Field(ge=1)
    long_unit_id: str
    score: float
    construction_policy: LongUnitConstructionPolicy

    member_doc_ids: tuple[str, ...]
    member_parent_ids: tuple[str, ...]
    member_chunk_ids: tuple[str, ...]


class ExpandedLongRAGDocument(ArtifactBaseModel):
    """One document expanded from retrieved long units."""

    query_id: str
    document_rank: int = Field(ge=1)
    doc_id: str

    source_long_unit_id: str
    source_long_unit_rank: int = Field(ge=1)
    source_long_unit_score: float

    membership_position: int = Field(ge=0)


class LongUnitConstructionSummary(ArtifactBaseModel):
    """Summary metrics for a constructed long-unit corpus."""

    dataset_id: str
    construction_policy: LongUnitConstructionPolicy
    target_token_size: int = Field(gt=0)

    source_segment_count: int = Field(ge=0)
    source_document_count: int = Field(ge=0)
    source_parent_count: int = Field(ge=0)

    long_unit_count: int = Field(ge=0)
    minimum_unit_tokens: int = Field(ge=0)
    maximum_unit_tokens: int = Field(ge=0)
    average_unit_tokens: float = Field(ge=0.0)

    complete_document_coverage: bool
    complete_parent_coverage: bool
    complete_chunk_coverage: bool


def build_long_unit_id(
    dataset_id: str,
    construction_policy: LongUnitConstructionPolicy,
    target_token_size: int,
    ordered_chunk_ids: Sequence[str],
) -> str:
    """Build a deterministic, content-addressable long unit ID."""
    clean_dataset = normalize_identifier(dataset_id)
    clean_policy = normalize_identifier(construction_policy)
    fingerprint_seed = ":".join(ordered_chunk_ids)
    fingerprint = short_fingerprint(fingerprint_seed, length=12)
    return (
        f"longunit-{clean_dataset}-{clean_policy}-"
        f"{target_token_size}-{fingerprint}"
    )



def _normalize_segment_order(
    segments: Sequence[LongRAGSourceSegment],
) -> tuple[LongRAGSourceSegment, ...]:
    """Return segments in deterministic canonical source order."""
    return tuple(
        sorted(
            segments,
            key=lambda s: (
                s.document_order,
                s.parent_order,
                s.chunk_order,
            ),
        )
    )


def construct_adjacent_long_units(
    dataset_id: str,
    segments: Sequence[LongRAGSourceSegment],
    target_token_size: int,
    token_counter: LongUnitTokenCounter | None = None,
) -> tuple[LongUnit, ...]:
    """Construct long units by greedily packing adjacent segments."""
    if not segments:
        return ()

    segments = _normalize_segment_order(segments)

    seen_chunks: set[str] = set()
    for segment in segments:
        if segment.chunk_id in seen_chunks:
            raise ValueError(
                f"Found duplicate chunk IDs: {segment.chunk_id}"
            )
        seen_chunks.add(segment.chunk_id)
        if segment.token_count > target_token_size:
            raise ValueError(
                f"Source segment {segment.chunk_id} token count "
                f"({segment.token_count}) exceeds target unit size "
                f"({target_token_size})."
            )

    units: list[LongUnit] = list()
    current_segments: list[LongRAGSourceSegment] = list()
    current_tokens = 0

    def flush_unit() -> None:
        if not current_segments:
            return
        chunk_ids = tuple(s.chunk_id for s in current_segments)
        unit_id = build_long_unit_id(
            dataset_id=dataset_id,
            construction_policy="adjacent_segments",
            target_token_size=target_token_size,
            ordered_chunk_ids=chunk_ids,
        )
        texts = [s.text for s in current_segments]
        combined_text = "\n\n".join(texts)
        if token_counter is not None:
            token_count = token_counter(combined_text)
        else:
            token_count = sum(s.token_count for s in current_segments)

        doc_ids = tuple(
            dict.fromkeys(s.doc_id for s in current_segments)
        )
        parent_ids = tuple(
            dict.fromkeys(s.parent_id for s in current_segments)
        )

        units.append(
            LongUnit(
                long_unit_id=unit_id,
                dataset_id=dataset_id,
                construction_policy="adjacent_segments",
                unit_order=len(units),
                text=combined_text,
                token_count=token_count,
                member_doc_ids=doc_ids,
                member_parent_ids=parent_ids,
                member_chunk_ids=chunk_ids,
                metadata={
                    "source_segment_count": len(current_segments),
                },
            )
        )
        current_segments.clear()

    for segment in segments:
        if (
            current_segments
            and current_tokens + segment.token_count > target_token_size
        ):
            flush_unit()
            current_tokens = 0
        current_segments.append(segment)
        current_tokens += segment.token_count

    flush_unit()
    return tuple(units)


def construct_parent_grouped_long_units(
    dataset_id: str,
    segments: Sequence[LongRAGSourceSegment],
    target_token_size: int,
    token_counter: LongUnitTokenCounter | None = None,
) -> tuple[LongUnit, ...]:
    """Construct long units grouped strictly by parent record boundaries."""
    if not segments:
        return ()

    segments = _normalize_segment_order(segments)

    seen_chunks: set[str] = set()
    parents_map: dict[tuple[str, str], list[LongRAGSourceSegment]] = {}

    for segment in segments:
        if segment.chunk_id in seen_chunks:
            raise ValueError(
                f"Found duplicate chunk IDs: {segment.chunk_id}"
            )
        seen_chunks.add(segment.chunk_id)
        if segment.token_count > target_token_size:
            raise ValueError(
                f"Source segment {segment.chunk_id} token count "
                f"({segment.token_count}) exceeds target unit size "
                f"({target_token_size})."
            )
        key = (segment.doc_id, segment.parent_id)
        parents_map.setdefault(key, []).append(segment)

    units: list[LongUnit] = list()

    for (doc_id, parent_id), parent_segments in parents_map.items():
        chunk_ids = tuple(s.chunk_id for s in parent_segments)
        unit_id = build_long_unit_id(
            dataset_id=dataset_id,
            construction_policy="structure_preserving_parent_groups",
            target_token_size=target_token_size,
            ordered_chunk_ids=chunk_ids,
        )
        texts = [s.text for s in parent_segments]
        combined_text = "\n\n".join(texts)
        if token_counter is not None:
            token_count = token_counter(combined_text)
        else:
            token_count = sum(s.token_count for s in parent_segments)

        doc_ids = (doc_id,)
        parent_ids = (parent_id,)

        units.append(
            LongUnit(
                long_unit_id=unit_id,
                dataset_id=dataset_id,
                construction_policy="structure_preserving_parent_groups",
                unit_order=len(units),
                text=combined_text,
                token_count=token_count,
                member_doc_ids=doc_ids,
                member_parent_ids=parent_ids,
                member_chunk_ids=chunk_ids,
                metadata={
                    "source_segment_count": len(parent_segments),
                },
            )
        )

    return tuple(units)


def construct_long_units(
    dataset_id: str,
    segments: Sequence[LongRAGSourceSegment],
    target_token_size: int,
    construction_policy: LongUnitConstructionPolicy = "adjacent_segments",
    token_counter: LongUnitTokenCounter | None = None,
) -> tuple[LongUnit, ...]:
    """Construct long units using the specified policy."""
    if construction_policy == "adjacent_segments":
        return construct_adjacent_long_units(
            dataset_id=dataset_id,
            segments=segments,
            target_token_size=target_token_size,
            token_counter=token_counter,
        )
    if construction_policy == "structure_preserving_parent_groups":
        return construct_parent_grouped_long_units(
            dataset_id=dataset_id,
            segments=segments,
            target_token_size=target_token_size,
            token_counter=token_counter,
        )
    raise ValueError(
        f"Unsupported construction policy: {construction_policy}"
    )


def summarize_long_units(
    dataset_id: str,
    construction_policy: LongUnitConstructionPolicy,
    target_token_size: int,
    source_segments: Sequence[LongRAGSourceSegment],
    long_units: Sequence[LongUnit],
) -> LongUnitConstructionSummary:
    """Summarize long-unit corpus construction metrics and coverage."""
    source_document_count = len(
        set(s.doc_id for s in source_segments)
    )
    source_parent_count = len(
        set((s.doc_id, s.parent_id) for s in source_segments)
    )

    unit_doc_ids = {
        doc_id
        for unit in long_units
        for doc_id in unit.member_doc_ids
    }
    unit_parent_ids = {
        parent_id
        for unit in long_units
        for parent_id in unit.member_parent_ids
    }
    unit_chunk_ids = {
        chunk_id
        for unit in long_units
        for chunk_id in unit.member_chunk_ids
    }

    source_chunk_ids = {s.chunk_id for s in source_segments}
    source_doc_ids = {s.doc_id for s in source_segments}
    source_parent_id_set = {s.parent_id for s in source_segments}

    complete_doc_cov = unit_doc_ids == source_doc_ids
    complete_parent_cov = unit_parent_ids == source_parent_id_set
    complete_chunk_cov = unit_chunk_ids == source_chunk_ids

    token_counts = [u.token_count for u in long_units]
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    avg_tokens = (
        sum(token_counts) / len(token_counts)
        if token_counts
        else 0.0
    )

    return LongUnitConstructionSummary(
        dataset_id=dataset_id,
        construction_policy=construction_policy,
        target_token_size=target_token_size,
        source_segment_count=len(source_segments),
        source_document_count=source_document_count,
        source_parent_count=source_parent_count,
        long_unit_count=len(long_units),
        minimum_unit_tokens=min_tokens,
        maximum_unit_tokens=max_tokens,
        average_unit_tokens=avg_tokens,
        complete_document_coverage=complete_doc_cov,
        complete_parent_coverage=complete_parent_cov,
        complete_chunk_coverage=complete_chunk_cov,
    )


def expand_ranked_long_units_to_documents(
    query_id: str,
    ranked_long_units: Sequence[RankedLongUnit],
    document_limit: int,
) -> tuple[ExpandedLongRAGDocument, ...]:
    """Expand retrieved long units into a deduplicated document ranking."""
    expanded: list[ExpandedLongRAGDocument] = list()
    seen_docs: set[str] = set()

    for unit in sorted(ranked_long_units, key=lambda u: u.rank):
        for position, doc_id in enumerate(unit.member_doc_ids):
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            expanded.append(
                ExpandedLongRAGDocument(
                    query_id=query_id,
                    document_rank=len(expanded) + 1,
                    doc_id=doc_id,
                    source_long_unit_id=unit.long_unit_id,
                    source_long_unit_rank=unit.rank,
                    source_long_unit_score=unit.score,
                    membership_position=position,
                )
            )
            if len(expanded) >= document_limit:
                return tuple(expanded)

    return tuple(expanded)
