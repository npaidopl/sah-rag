"""Parent-level aggregation of retrieved child chunks."""

from __future__ import annotations

from collections import defaultdict

from rag_system.retrieval.models import (
    RetrievedChunk,
    RetrievedParent,
)


def aggregate_chunks_to_parents(
    chunks: tuple[RetrievedChunk, ...],
    *,
    additional_child_weight: float = 0.05,
    limit: int | None = None,
) -> tuple[RetrievedParent, ...]:
    """Aggregate ranked child evidence into parent results."""
    if additional_child_weight < 0:
        raise ValueError(
            "additional_child_weight cannot be negative."
        )

    if limit is not None and limit < 1:
        raise ValueError(
            "Parent result limit must be positive."
        )

    grouped: dict[str, list[RetrievedChunk]] = defaultdict(
        list
    )

    for chunk in chunks:
        grouped[chunk.parent_id].append(chunk)

    provisional: list[
        tuple[float, int, str, list[RetrievedChunk]]
    ] = []

    for parent_id, parent_chunks in grouped.items():
        ordered = sorted(
            parent_chunks,
            key=lambda chunk: (
                chunk.rank,
                -chunk.score,
                chunk.chunk_id,
            ),
        )

        score_ordered = sorted(
            parent_chunks,
            key=lambda chunk: (
                -chunk.score,
                chunk.rank,
                chunk.chunk_id,
            ),
        )

        best_score = score_ordered[0].score
        additional_score = sum(
            chunk.score
            for chunk in score_ordered[1:]
        )

        parent_score = (
            best_score
            + additional_child_weight
            * additional_score
        )

        provisional.append(
            (
                parent_score,
                ordered[0].rank,
                parent_id,
                ordered,
            )
        )

    provisional.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2],
        )
    )

    if limit is not None:
        provisional = provisional[:limit]

    parents: list[RetrievedParent] = []

    for parent_rank, (
        parent_score,
        best_chunk_rank,
        parent_id,
        evidence,
    ) in enumerate(provisional, start=1):
        first = evidence[0]

        parents.append(
            RetrievedParent(
                parent_id=parent_id,
                doc_id=first.doc_id,
                dataset=first.dataset,
                rank=parent_rank,
                score=parent_score,
                best_chunk_rank=best_chunk_rank,
                matched_chunk_count=len(evidence),
                chunk_ids=tuple(
                    chunk.chunk_id
                    for chunk in evidence
                ),
                child_scores=tuple(
                    chunk.score
                    for chunk in evidence
                ),
            )
        )

    return tuple(parents)
