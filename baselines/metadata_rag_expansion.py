"""Canonical expansion for the Metadata-RAG comparator.

Per amendment 04, one canonical document may be split into one or
more windows. Each window is a retrieval item with its own rank and
its own fused score. The document expansion stage converts the
window ranking into a canonical-document ranking by first-occurrence
deduplication:

  * Iterate windows in ascending rank.
  * Emit the canonical document of each window if it has not been
    emitted yet.
  * Skip subsequent windows of an already-emitted document.
  * Preserve the originating window's rank and scores on the emitted
    document.

The method matches the LongRAG document expansion precedent exactly.
It never reads ground truth and never reshapes or reorders the
surviving document ranking.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from baselines.metadata_rag_retrieval import (
    RankedItem,
)

EXPANSION_VERSION: Final[str] = "0.2.0"

_MEMBERSHIP_POSITION: Final[int] = 0


class MetadataRAGExpansionError(ValueError):
    """Raised when canonical expansion violates its contract."""


class MetadataRAGBaseModel(BaseModel):
    """Strict, frozen base model for expansion artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )


class ExpandedDocument(MetadataRAGBaseModel):
    """One canonical document produced by the expansion stage.

    Fields:
        query_id: canonical query identifier.
        document_rank: 1-based rank within the deduplicated document
            ranking. Independent of the window ranks.
        doc_id: canonical document identifier (== lineage_doc_id).
        source_item_rank: rank of the originating window.
        source_item_score: fused score of the originating window.
        source_item_fused_score: equal to source_item_score; kept as
            a separate field for contract symmetry.
        source_item_semantic_score: semantic component of the
            originating window.
        source_item_metadata_score: metadata component of the
            originating window.
        membership_position: always 0, because each surviving window
            maps to exactly one document.
    """

    query_id: str = Field(min_length=1)
    document_rank: int = Field(ge=1)
    doc_id: str = Field(min_length=1)
    source_item_rank: int = Field(ge=1)
    source_item_score: float
    source_item_fused_score: float
    source_item_semantic_score: float
    source_item_metadata_score: float
    membership_position: int = Field(ge=0)

    @model_validator(mode="after")
    def _enforce_finite(self) -> ExpandedDocument:
        for name, value in (
            ("source_item_score", self.source_item_score),
            ("source_item_fused_score", self.source_item_fused_score),
            (
                "source_item_semantic_score",
                self.source_item_semantic_score,
            ),
            (
                "source_item_metadata_score",
                self.source_item_metadata_score,
            ),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.membership_position != _MEMBERSHIP_POSITION:
            raise ValueError(
                "membership_position must be 0 for Metadata-RAG"
            )
        return self


def expand_ranked_items_to_documents(
    *,
    query_id: str,
    ranked_items: Sequence[RankedItem],
    document_limit: int,
) -> tuple[ExpandedDocument, ...]:
    """Convert a ranked-item sequence into a document ranking.

    The item-to-document mapping is many-to-one under windowing, so
    the expansion deduplicates by ``canonical_doc_id`` using the
    first-occurrence rule: the highest-ranked window of each document
    determines that document's emitted rank and scores.

    Args:
        query_id: canonical query identifier.
        ranked_items: ordered sequence of RankedItems, ranks starting
            at 1 and contiguous.
        document_limit: maximum number of documents to emit (>= 1).

    Returns:
        A tuple of ExpandedDocument records, document ranks starting
        at 1 and contiguous up to ``min(document_limit, n_unique)``.
    """
    if not query_id.strip():
        raise MetadataRAGExpansionError("query_id cannot be empty")
    if document_limit < 1:
        raise MetadataRAGExpansionError(
            "document_limit must be >= 1"
        )

    items = tuple(ranked_items)
    expected_ranks = list(range(1, len(items) + 1))
    observed_ranks = [item.rank for item in items]
    if observed_ranks != expected_ranks:
        raise MetadataRAGExpansionError(
            "ranked_items ranks must be contiguous starting at 1; "
            f"observed {observed_ranks}"
        )

    seen_doc_ids: set[str] = set()
    documents: list[ExpandedDocument] = []
    for item in items:
        if item.canonical_doc_id in seen_doc_ids:
            continue
        if len(documents) >= document_limit:
            break
        seen_doc_ids.add(item.canonical_doc_id)
        documents.append(
            ExpandedDocument(
                query_id=query_id,
                document_rank=len(documents) + 1,
                doc_id=item.canonical_doc_id,
                source_item_rank=item.rank,
                source_item_score=item.score,
                source_item_fused_score=item.metadata.fused_score,
                source_item_semantic_score=item.metadata.semantic_score,
                source_item_metadata_score=item.metadata.metadata_score,
                membership_position=_MEMBERSHIP_POSITION,
            )
        )

    return tuple(documents)
