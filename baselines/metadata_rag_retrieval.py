"""Online retrieval pipeline for the Metadata-RAG comparator.

Implements the frozen online path:

    query_text
      -> MetadataRecord       (build_query_metadata)
      -> constraints          (derive_query_constraints)
      -> candidate windows    (min_match_ratio = 0.5)
      -> semantic search      (ReferenceDenseIndex, exact cosine)
      -> fused ranking        (alpha_metadata 0.3, alpha_semantic 0.7)
      -> ranked items         (retrieval_boundary output contract)

Per amendment 04, one canonical document may be split into one or
more windows. Each window is a retrieval item with its own metadata
record and its own dense vector. The retrieval pipeline reads and
fuses scores per window; canonical identity is exposed on every item
as ``canonical_doc_id`` (equal to the window's ``lineage_doc_id``).

The pipeline reads only query_id and query_text. It never reads
doc_ids, relevant_ids, answers, target_content_group_id, or any
other ground-truth surface.

The fallback when zero windows pass the filter is to use the full
corpus for semantic retrieval with alpha_metadata = 0.0 and to flag
metadata_match_none = true on every emitted item.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from baselines.metadata_rag_adapter import (
    MetadataRAGProjectedQuery,
)
from baselines.metadata_rag_metadata import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    MetadataConstraint,
    MetadataRecord,
    build_query_metadata,
    derive_query_constraints,
)
from baselines.metadata_rag_reference_index import (
    ReferenceDenseIndex,
)

RETRIEVAL_VERSION: Final[str] = "0.2.0"

FROZEN_ALPHA_METADATA: Final[float] = 0.3
FROZEN_ALPHA_SEMANTIC: Final[float] = 0.7
FROZEN_MIN_MATCH_RATIO: Final[float] = 0.5
FROZEN_FALLBACK_ALPHA_METADATA: Final[float] = 0.0
DEFAULT_DOCUMENT_LIMIT: Final[int] = 100

_TOLERANCE: Final[float] = 1e-9


class MetadataRAGRetrievalError(ValueError):
    """Raised when the retrieval pipeline violates its frozen contract."""


class MetadataRAGBaseModel(BaseModel):
    """Strict, frozen base model for retrieval artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )


class RetrievalConfig(MetadataRAGBaseModel):
    """Frozen retrieval configuration.

    ``alpha_metadata``, ``alpha_semantic`` and ``min_match_ratio`` are
    architecture-defining thresholds frozen by the clean-room spec.
    They are exposed here only so that a run can carry them; any
    deviation from the frozen values is rejected at construction.
    """

    document_limit: int = Field(default=DEFAULT_DOCUMENT_LIMIT, ge=1)
    alpha_metadata: float = FROZEN_ALPHA_METADATA
    alpha_semantic: float = FROZEN_ALPHA_SEMANTIC
    min_match_ratio: float = FROZEN_MIN_MATCH_RATIO

    @model_validator(mode="after")
    def _enforce_frozen_constants(self) -> RetrievalConfig:
        if not math.isclose(
            self.alpha_metadata,
            FROZEN_ALPHA_METADATA,
            abs_tol=_TOLERANCE,
        ):
            raise ValueError(
                "alpha_metadata is frozen at "
                f"{FROZEN_ALPHA_METADATA}"
            )
        if not math.isclose(
            self.alpha_semantic,
            FROZEN_ALPHA_SEMANTIC,
            abs_tol=_TOLERANCE,
        ):
            raise ValueError(
                "alpha_semantic is frozen at "
                f"{FROZEN_ALPHA_SEMANTIC}"
            )
        if not math.isclose(
            self.min_match_ratio,
            FROZEN_MIN_MATCH_RATIO,
            abs_tol=_TOLERANCE,
        ):
            raise ValueError(
                "min_match_ratio is frozen at "
                f"{FROZEN_MIN_MATCH_RATIO}"
            )
        if not math.isclose(
            self.alpha_metadata + self.alpha_semantic,
            1.0,
            abs_tol=_TOLERANCE,
        ):
            raise ValueError(
                "alpha_metadata + alpha_semantic must equal 1.0"
            )
        return self


class ConstraintMatch(MetadataRAGBaseModel):
    """One constraint that a specific window satisfied."""

    constraint_id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    value: str = Field(min_length=1)


class FallbackFlags(MetadataRAGBaseModel):
    """Per-item flags describing fallback conditions."""

    metadata_match_none: bool
    metadata_missing_on_document: bool


class RankedItemMetadata(MetadataRAGBaseModel):
    """Per-item metadata exposed by the retrieval boundary.

    All scores are finite. ``fused_score`` is the primary item score
    and equals ``score`` on the parent RankedItem.
    """

    semantic_score: float
    metadata_score: float
    fused_score: float
    metadata_match_evidence: tuple[ConstraintMatch, ...]
    fallback_flags: FallbackFlags

    @model_validator(mode="after")
    def _enforce_finite(self) -> RankedItemMetadata:
        for name, value in (
            ("semantic_score", self.semantic_score),
            ("metadata_score", self.metadata_score),
            ("fused_score", self.fused_score),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return self


class RankedItem(MetadataRAGBaseModel):
    """One ranked window returned by the retrieval pipeline.

    ``architecture_item_id`` is the window identifier. ``canonical_doc_id``
    is the window's ``lineage_doc_id`` (canonical document identity).
    ``canonical_chunk_or_evidence_ids`` carries the deterministic chunk
    identifier of the window.
    """

    architecture_item_id: str = Field(min_length=1)
    canonical_doc_id: str = Field(min_length=1)
    canonical_chunk_or_evidence_ids: tuple[str, ...]
    rank: int = Field(ge=1)
    score: float
    score_semantics: Literal["HIGHER_IS_BETTER"] = "HIGHER_IS_BETTER"
    metadata: RankedItemMetadata

    @model_validator(mode="after")
    def _enforce_shape(self) -> RankedItem:
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if not self.canonical_chunk_or_evidence_ids:
            raise ValueError(
                "canonical_chunk_or_evidence_ids must be non-empty"
            )
        return self


class RetrievalRunMetadata(MetadataRAGBaseModel):
    """Top-level metadata for one query's retrieval run."""

    alpha_metadata: float
    alpha_semantic: float
    schema_version: str = SCHEMA_VERSION
    extractor_version: str = EXTRACTOR_VERSION
    fallback_used: bool
    constraint_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    retrieval_version: str = RETRIEVAL_VERSION


class QueryRetrievalResult(MetadataRAGBaseModel):
    """Complete retrieval output for a single query."""

    query_id: str = Field(min_length=1)
    ranked_items: tuple[RankedItem, ...]
    metadata: RetrievalRunMetadata


# --- Internals --------------------------------------------------------------


def _constraint_matches(
    constraint: MetadataConstraint, metadata: MetadataRecord
) -> bool:
    """Apply the frozen per-field comparison semantics.

    entity (capitalized_phrases): case-insensitive exact
    temporal (year_mentions): integer equality
    numeric (numeric_tokens): string equality of normalized form
    """
    if constraint.field == "capitalized_phrases":
        return constraint.value in metadata.capitalized_phrases
    if constraint.field == "year_mentions":
        return any(
            str(year) == constraint.value
            for year in metadata.year_mentions
        )
    if constraint.field == "numeric_tokens":
        return constraint.value in metadata.numeric_tokens
    return False


def _is_metadata_missing(
    metadata: MetadataRecord,
    constraints: tuple[MetadataConstraint, ...],
) -> bool:
    """True when the window lacks the field any constraint refers to.

    A field with an empty tuple counts as missing for this flag.
    """
    for constraint in constraints:
        if (
            constraint.field == "capitalized_phrases"
            and not metadata.capitalized_phrases
        ):
            return True
        if (
            constraint.field == "year_mentions"
            and not metadata.year_mentions
        ):
            return True
        if (
            constraint.field == "numeric_tokens"
            and not metadata.numeric_tokens
        ):
            return True
    return False


def _build_result(
    *,
    query_id: str,
    ranked_items: list[RankedItem],
    alpha_metadata: float,
    alpha_semantic: float,
    fallback_used: bool,
    constraint_count: int,
    candidate_count: int,
) -> QueryRetrievalResult:
    return QueryRetrievalResult(
        query_id=query_id,
        ranked_items=tuple(ranked_items),
        metadata=RetrievalRunMetadata(
            alpha_metadata=alpha_metadata,
            alpha_semantic=alpha_semantic,
            fallback_used=fallback_used,
            constraint_count=constraint_count,
            candidate_count=candidate_count,
        ),
    )


# --- Public pipeline --------------------------------------------------------


def retrieve_query(
    *,
    query: MetadataRAGProjectedQuery,
    index: ReferenceDenseIndex,
    query_vector: Sequence[float],
    config: RetrievalConfig,
) -> QueryRetrievalResult:
    """Run the full retrieval pipeline for one query.

    The pipeline reads only ``query.query_id`` and ``query.query_text``.
    It derives constraints from the query text, applies the frozen
    candidate filter over the window metadata records, performs exact
    dense cosine search over the candidate windows, fuses metadata and
    semantic scores with the frozen coefficients, and returns a ranked
    list of windows. Each emitted item exposes both the window identity
    and the canonical document identity.

    Args:
        query: a projected query carrying query_id and query_text.
        index: the frozen exact dense reference index.
        query_vector: L2-normalized dense vector of the query, of the
            same dimension as the index.
        config: frozen retrieval configuration.

    Returns:
        A QueryRetrievalResult whose ranked_items carry the full
        per-item evidence required by the retrieval boundary.
    """
    if not isinstance(query, MetadataRAGProjectedQuery):
        raise TypeError(
            "query must be a MetadataRAGProjectedQuery instance"
        )
    if not isinstance(index, ReferenceDenseIndex):
        raise TypeError(
            "index must be a ReferenceDenseIndex instance"
        )
    if not isinstance(config, RetrievalConfig):
        raise TypeError(
            "config must be a RetrievalConfig instance"
        )

    query_record = build_query_metadata(
        dataset=index.dataset_id,
        query_id=query.query_id,
        query_text=query.query_text,
    )
    constraints = derive_query_constraints(query_record)

    entries = index.entries()
    match_evidence_by_window: dict[str, tuple[ConstraintMatch, ...]] = {}
    match_count_by_window: dict[str, int] = {}
    missing_by_window: dict[str, bool] = {}

    for entry in entries:
        matched: list[ConstraintMatch] = []
        for constraint in constraints:
            if _constraint_matches(constraint, entry.metadata):
                matched.append(
                    ConstraintMatch(
                        constraint_id=constraint.constraint_id,
                        field=constraint.field,
                        value=constraint.value,
                    )
                )
        match_evidence_by_window[entry.doc_id] = tuple(matched)
        match_count_by_window[entry.doc_id] = len(matched)
        missing_by_window[entry.doc_id] = _is_metadata_missing(
            entry.metadata, constraints
        )

    total_constraints = max(1, len(constraints))
    match_ratio_by_window: dict[str, float] = {
        doc_id: match_count_by_window[doc_id] / total_constraints
        for doc_id in match_evidence_by_window
    }

    if constraints:
        candidate_doc_ids = frozenset(
            doc_id
            for doc_id, ratio in match_ratio_by_window.items()
            if ratio >= config.min_match_ratio
        )
    else:
        candidate_doc_ids = frozenset()

    fallback_used = len(candidate_doc_ids) == 0
    if fallback_used:
        candidate_doc_ids = frozenset(
            match_evidence_by_window.keys()
        )
        effective_alpha_metadata = FROZEN_FALLBACK_ALPHA_METADATA
    else:
        effective_alpha_metadata = config.alpha_metadata
    effective_alpha_semantic = 1.0 - effective_alpha_metadata

    if not candidate_doc_ids:
        return _build_result(
            query_id=query.query_id,
            ranked_items=[],
            alpha_metadata=effective_alpha_metadata,
            alpha_semantic=effective_alpha_semantic,
            fallback_used=fallback_used,
            constraint_count=len(constraints),
            candidate_count=0,
        )

    hits = index.search(
        query_vector,
        top_k=len(candidate_doc_ids),
        candidate_doc_ids=candidate_doc_ids,
    )

    scored: list[tuple[float, float, str]] = []
    for hit in hits:
        metadata_score = (
            0.0
            if fallback_used
            else match_ratio_by_window[hit.doc_id]
        )
        fused_score = (
            effective_alpha_metadata * metadata_score
            + effective_alpha_semantic * float(hit.score)
        )
        scored.append(
            (fused_score, float(hit.score), hit.doc_id)
        )

    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))

    limit = min(config.document_limit, len(scored))
    ranked_items: list[RankedItem] = []
    for rank, (fused_score, semantic_score, window_doc_id) in enumerate(
        scored[:limit], start=1
    ):
        # Look up the originating hit to recover its lineage and
        # chunk id deterministically.
        matching_hit = next(
            h for h in hits if h.doc_id == window_doc_id
        )
        metadata_score = (
            0.0
            if fallback_used
            else match_ratio_by_window[window_doc_id]
        )
        ranked_items.append(
            RankedItem(
                architecture_item_id=window_doc_id,
                canonical_doc_id=matching_hit.lineage_doc_id,
                canonical_chunk_or_evidence_ids=(
                    matching_hit.chunk_id,
                ),
                rank=rank,
                score=fused_score,
                metadata=RankedItemMetadata(
                    semantic_score=semantic_score,
                    metadata_score=metadata_score,
                    fused_score=fused_score,
                    metadata_match_evidence=match_evidence_by_window[
                        window_doc_id
                    ],
                    fallback_flags=FallbackFlags(
                        metadata_match_none=fallback_used,
                        metadata_missing_on_document=missing_by_window[
                            window_doc_id
                        ],
                    ),
                ),
            )
        )

    return _build_result(
        query_id=query.query_id,
        ranked_items=ranked_items,
        alpha_metadata=effective_alpha_metadata,
        alpha_semantic=effective_alpha_semantic,
        fallback_used=fallback_used,
        constraint_count=len(constraints),
        candidate_count=len(candidate_doc_ids),
    )


def retrieve_batch(
    *,
    queries: Sequence[MetadataRAGProjectedQuery],
    index: ReferenceDenseIndex,
    query_vectors: Sequence[Sequence[float]],
    config: RetrievalConfig,
) -> tuple[QueryRetrievalResult, ...]:
    """Run the retrieval pipeline over a batch of queries.

    ``queries`` and ``query_vectors`` must have identical length and
    be aligned positionally. The order of the returned results matches
    the order of the inputs.
    """
    if len(queries) != len(query_vectors):
        raise MetadataRAGRetrievalError(
            "queries and query_vectors must have identical length; "
            f"observed {len(queries)} and {len(query_vectors)}"
        )
    return tuple(
        retrieve_query(
            query=q,
            index=index,
            query_vector=v,
            config=config,
        )
        for q, v in zip(queries, query_vectors, strict=True)
    )
