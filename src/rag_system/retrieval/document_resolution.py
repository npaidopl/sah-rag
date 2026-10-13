"""Typed canonical-document resolution from ranked parents."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import Field, model_validator

from rag_system.artifacts import ArtifactBaseModel
from rag_system.retrieval.models import RetrievedParent

PROHIBITED_METADATA_KEYS = frozenset(
    {
        "generated_answer",
        "reference_answer",
        "answer_text",
        "relevance",
        "relevance_label",
        "judgment",
        "evaluation_metric",
        "expected_group",
        "is_relevant",
        "hit",
        "metric",
    }
)


class RetrievedDocument(ArtifactBaseModel):
    """A canonical-document ranking derived from ranked parents."""

    query_id: str
    doc_id: str
    dataset: str
    rank: int = Field(ge=1)
    score: float
    best_parent_rank: int = Field(ge=1)
    contributing_parent_ids: tuple[str, ...]
    contributing_chunk_ids: tuple[str, ...]
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_document(self) -> RetrievedDocument:
        if not self.query_id.strip():
            raise ValueError("query_id cannot be empty.")
        if not self.doc_id.strip():
            raise ValueError("doc_id cannot be empty.")
        if not self.dataset.strip():
            raise ValueError("dataset cannot be empty.")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite.")
        if len(self.contributing_parent_ids) != len(set(self.contributing_parent_ids)):
            raise ValueError("contributing_parent_ids must be unique.")
        if len(self.contributing_chunk_ids) != len(set(self.contributing_chunk_ids)):
            raise ValueError("contributing_chunk_ids must be unique.")
        forbidden = PROHIBITED_METADATA_KEYS & set(self.metadata.keys())
        if forbidden:
            raise ValueError(
                "metadata contains prohibited field(s): "
                + ", ".join(sorted(forbidden))
            )
        return self


def resolve_parents_to_documents(
    parents: Sequence[RetrievedParent],
    *,
    query_id: str,
    limit: int | None = None,
) -> tuple[RetrievedDocument, ...]:
    """Resolve ranked parents to canonical-document rankings.

    Grouping is by ``doc_id``. The document score is the score of
    the highest-ranked contributing parent. Duplicate-content
    documents are not deduplicated. Ties are broken deterministically
    by ``best_parent_rank`` and ``doc_id``.
    """
    if limit is not None and limit < 1:
        raise ValueError("Document result limit must be positive.")

    if not parents:
        return ()

    grouped: dict[str, list[RetrievedParent]] = {}
    for parent in parents:
        grouped.setdefault(parent.doc_id, []).append(parent)

    provisional: list[
        tuple[float, int, str, str, tuple[str, ...], tuple[str, ...]]
    ] = []

    for doc_id, doc_parents in grouped.items():
        ordered_parents = sorted(
            doc_parents,
            key=lambda p: (p.rank, p.parent_id),
        )
        score_ordered = sorted(
            doc_parents,
            key=lambda p: (-p.score, p.rank, p.parent_id),
        )
        best_parent = score_ordered[0]
        document_score = best_parent.score
        best_parent_rank = ordered_parents[0].rank
        dataset = best_parent.dataset

        parent_ids: list[str] = []
        chunk_ids: list[str] = []
        seen_parents: set[str] = set()
        seen_chunks: set[str] = set()
        for p in ordered_parents:
            if p.parent_id not in seen_parents:
                seen_parents.add(p.parent_id)
                parent_ids.append(p.parent_id)
            for cid in p.chunk_ids:
                if cid not in seen_chunks:
                    seen_chunks.add(cid)
                    chunk_ids.append(cid)

        provisional.append(
            (
                document_score,
                best_parent_rank,
                doc_id,
                dataset,
                tuple(parent_ids),
                tuple(chunk_ids),
            )
        )

    provisional.sort(
        key=lambda item: (-item[0], item[1], item[2])
    )

    if limit is not None:
        provisional = provisional[:limit]

    results: list[RetrievedDocument] = []
    for rank, (
        document_score,
        best_parent_rank,
        doc_id,
        dataset,
        parent_ids_t,
        chunk_ids_t,
    ) in enumerate(provisional, start=1):
        results.append(
            RetrievedDocument(
                query_id=query_id,
                doc_id=doc_id,
                dataset=dataset,
                rank=rank,
                score=document_score,
                best_parent_rank=best_parent_rank,
                contributing_parent_ids=parent_ids_t,
                contributing_chunk_ids=chunk_ids_t,
                metadata={},
            )
        )

    return tuple(results)