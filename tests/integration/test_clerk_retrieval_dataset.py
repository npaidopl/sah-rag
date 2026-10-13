"""Integration tests for the canonical Clerk benchmark."""

from __future__ import annotations

from rag_system.evaluation import (
    load_clerk_retrieval_dataset,
)


def test_clerk_retrieval_dataset_is_fully_mapped() -> None:
    """All Clerk benchmark queries and qrels must align."""
    dataset = load_clerk_retrieval_dataset()

    assert len(dataset.queries) == 1000
    assert len(dataset.qrels) == 1000
    assert len(dataset.document_ids) == 2756

    for query in dataset.queries:
        assert query.query_id in dataset.qrels

    positive_ids = {
        judgment.item_id
        for judgment in dataset.qrels[
            query.query_id
        ]
        if judgment.relevance > 0
    }

    assert positive_ids == set(
        query.gold_passage_ids
    )

    assert positive_ids.issubset(
        dataset.document_ids
    )
