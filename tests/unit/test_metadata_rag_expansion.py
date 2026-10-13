"""Tests for rag_system.baselines.metadata_rag_expansion."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from rag_system.baselines.metadata_rag_expansion import (
    EXPANSION_VERSION,
    ExpandedDocument,
    MetadataRAGExpansionError,
    expand_ranked_items_to_documents,
)
from rag_system.baselines.metadata_rag_retrieval import (
    FallbackFlags,
    RankedItem,
    RankedItemMetadata,
)


def _item(
    *,
    architecture_item_id: str | None = None,
    canonical_doc_id: str,
    chunk_id: str,
    rank: int,
    fused: float = 0.9,
    semantic: float = 0.8,
    meta: float = 0.5,
) -> RankedItem:
    arch_id = (
        architecture_item_id
        if architecture_item_id is not None
        else canonical_doc_id
    )
    return RankedItem(
        architecture_item_id=arch_id,
        canonical_doc_id=canonical_doc_id,
        canonical_chunk_or_evidence_ids=(chunk_id,),
        rank=rank,
        score=fused,
        metadata=RankedItemMetadata(
            semantic_score=semantic,
            metadata_score=meta,
            fused_score=fused,
            metadata_match_evidence=(),
            fallback_flags=FallbackFlags(
                metadata_match_none=False,
                metadata_missing_on_document=False,
            ),
        ),
    )


# --- Successful expansion --------------------------------------------------


def test_expand_single_item() -> None:
    item = _item(
        canonical_doc_id="d-1",
        chunk_id="d-1:c0",
        rank=1,
        fused=0.95,
    )
    docs = expand_ranked_items_to_documents(
        query_id="q-1",
        ranked_items=(item,),
        document_limit=10,
    )
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, ExpandedDocument)
    assert doc.query_id == "q-1"
    assert doc.document_rank == 1
    assert doc.doc_id == "d-1"
    assert doc.source_item_rank == 1
    assert doc.source_item_score == 0.95
    assert doc.source_item_fused_score == 0.95
    assert doc.membership_position == 0


def test_expand_preserves_item_order_and_ranks() -> None:
    items = (
        _item(canonical_doc_id="d-a", chunk_id="d-a:c0", rank=1),
        _item(canonical_doc_id="d-b", chunk_id="d-b:c0", rank=2),
        _item(canonical_doc_id="d-c", chunk_id="d-c:c0", rank=3),
    )
    docs = expand_ranked_items_to_documents(
        query_id="q",
        ranked_items=items,
        document_limit=100,
    )
    assert [d.doc_id for d in docs] == ["d-a", "d-b", "d-c"]
    assert [d.document_rank for d in docs] == [1, 2, 3]
    assert [d.source_item_rank for d in docs] == [1, 2, 3]


def test_expand_preserves_scores_exactly() -> None:
    items = (
        _item(
            canonical_doc_id="d-a",
            chunk_id="d-a:c0",
            rank=1,
            fused=0.8765,
            semantic=0.6543,
            meta=0.4321,
        ),
    )
    docs = expand_ranked_items_to_documents(
        query_id="q",
        ranked_items=items,
        document_limit=10,
    )
    d = docs[0]
    assert math.isclose(d.source_item_score, 0.8765, abs_tol=1e-9)
    assert math.isclose(
        d.source_item_fused_score, 0.8765, abs_tol=1e-9
    )
    assert math.isclose(
        d.source_item_semantic_score, 0.6543, abs_tol=1e-9
    )
    assert math.isclose(
        d.source_item_metadata_score, 0.4321, abs_tol=1e-9
    )


def test_expand_with_document_limit_truncates() -> None:
    items = tuple(
        _item(
            canonical_doc_id=f"d-{i}",
            chunk_id=f"d-{i}:c0",
            rank=i,
        )
        for i in range(1, 11)
    )
    docs = expand_ranked_items_to_documents(
        query_id="q",
        ranked_items=items,
        document_limit=3,
    )
    assert len(docs) == 3
    assert [d.doc_id for d in docs] == ["d-1", "d-2", "d-3"]


def test_expand_empty_items_returns_empty() -> None:
    docs = expand_ranked_items_to_documents(
        query_id="q",
        ranked_items=(),
        document_limit=100,
    )
    assert docs == ()


# --- Deduplication of multi-window documents ------------------------------


def test_expand_dedupes_duplicate_canonical_doc_ids() -> None:
    # Three windows: two belong to d1, one to d2. Order is d1:w0,
    # d1:w1, d2:w0. The output must be d1 (from the first window),
    # then d2.
    items = (
        _item(
            architecture_item_id="d1:w0",
            canonical_doc_id="d1",
            chunk_id="d1:c0",
            rank=1,
            fused=0.9,
        ),
        _item(
            architecture_item_id="d1:w1",
            canonical_doc_id="d1",
            chunk_id="d1:c1",
            rank=2,
            fused=0.8,
        ),
        _item(
            architecture_item_id="d2:w0",
            canonical_doc_id="d2",
            chunk_id="d2:c0",
            rank=3,
            fused=0.7,
        ),
    )
    docs = expand_ranked_items_to_documents(
        query_id="q",
        ranked_items=items,
        document_limit=100,
    )
    assert [d.doc_id for d in docs] == ["d1", "d2"]
    assert [d.document_rank for d in docs] == [1, 2]
    # First occurrence of d1 determines its score.
    assert docs[0].source_item_rank == 1
    assert math.isclose(
        docs[0].source_item_fused_score, 0.9, abs_tol=1e-9
    )


def test_expand_dedupes_interleaved_windows() -> None:
    # Windows interleave across two documents.
    items = (
        _item(
            architecture_item_id="d1:w0",
            canonical_doc_id="d1",
            chunk_id="d1:c0",
            rank=1,
            fused=0.9,
        ),
        _item(
            architecture_item_id="d2:w0",
            canonical_doc_id="d2",
            chunk_id="d2:c0",
            rank=2,
            fused=0.85,
        ),
        _item(
            architecture_item_id="d1:w1",
            canonical_doc_id="d1",
            chunk_id="d1:c1",
            rank=3,
            fused=0.8,
        ),
        _item(
            architecture_item_id="d2:w1",
            canonical_doc_id="d2",
            chunk_id="d2:c1",
            rank=4,
            fused=0.75,
        ),
    )
    docs = expand_ranked_items_to_documents(
        query_id="q",
        ranked_items=items,
        document_limit=100,
    )
    assert [d.doc_id for d in docs] == ["d1", "d2"]
    assert docs[0].source_item_rank == 1
    assert docs[1].source_item_rank == 2


def test_expand_dedupe_respects_document_limit() -> None:
    # 6 windows of 3 documents each, document_limit=2 -> two docs.
    items = (
        _item(
            architecture_item_id="d1:w0",
            canonical_doc_id="d1",
            chunk_id="d1:c0",
            rank=1,
        ),
        _item(
            architecture_item_id="d1:w1",
            canonical_doc_id="d1",
            chunk_id="d1:c1",
            rank=2,
        ),
        _item(
            architecture_item_id="d2:w0",
            canonical_doc_id="d2",
            chunk_id="d2:c0",
            rank=3,
        ),
        _item(
            architecture_item_id="d2:w1",
            canonical_doc_id="d2",
            chunk_id="d2:c1",
            rank=4,
        ),
        _item(
            architecture_item_id="d3:w0",
            canonical_doc_id="d3",
            chunk_id="d3:c0",
            rank=5,
        ),
        _item(
            architecture_item_id="d3:w1",
            canonical_doc_id="d3",
            chunk_id="d3:c1",
            rank=6,
        ),
    )
    docs = expand_ranked_items_to_documents(
        query_id="q",
        ranked_items=items,
        document_limit=2,
    )
    assert [d.doc_id for d in docs] == ["d1", "d2"]


def test_expand_duplicate_content_but_distinct_doc_ids_both_emitted() -> None:
    # Two different canonical documents with equal scores are distinct.
    items = (
        _item(
            canonical_doc_id="pilot-doc-01",
            chunk_id="pilot-doc-01:c0",
            rank=1,
            fused=0.9,
        ),
        _item(
            canonical_doc_id="pilot-doc-02",
            chunk_id="pilot-doc-02:c0",
            rank=2,
            fused=0.9,
        ),
    )
    docs = expand_ranked_items_to_documents(
        query_id="q",
        ranked_items=items,
        document_limit=10,
    )
    assert [d.doc_id for d in docs] == ["pilot-doc-01", "pilot-doc-02"]


# --- Validation ------------------------------------------------------------


def test_expand_rejects_empty_query_id() -> None:
    with pytest.raises(MetadataRAGExpansionError, match="query_id"):
        expand_ranked_items_to_documents(
            query_id="",
            ranked_items=(
                _item(
                    canonical_doc_id="d", chunk_id="d:c0", rank=1
                ),
            ),
            document_limit=10,
        )


def test_expand_rejects_zero_document_limit() -> None:
    with pytest.raises(
        MetadataRAGExpansionError, match="document_limit"
    ):
        expand_ranked_items_to_documents(
            query_id="q",
            ranked_items=(
                _item(
                    canonical_doc_id="d", chunk_id="d:c0", rank=1
                ),
            ),
            document_limit=0,
        )


def test_expand_rejects_non_contiguous_ranks() -> None:
    items = (
        _item(canonical_doc_id="d-a", chunk_id="d-a:c0", rank=1),
        _item(canonical_doc_id="d-b", chunk_id="d-b:c0", rank=3),
    )
    with pytest.raises(
        MetadataRAGExpansionError, match="contiguous"
    ):
        expand_ranked_items_to_documents(
            query_id="q",
            ranked_items=items,
            document_limit=10,
        )


# --- Model ----------------------------------------------------------------


def test_expanded_document_is_frozen() -> None:
    docs = expand_ranked_items_to_documents(
        query_id="q",
        ranked_items=(
            _item(canonical_doc_id="d", chunk_id="d:c0", rank=1),
        ),
        document_limit=10,
    )
    with pytest.raises(ValidationError):
        docs[0].doc_id = "changed"


def test_expanded_document_rejects_nonzero_membership_position() -> None:
    with pytest.raises(ValidationError, match="membership_position"):
        ExpandedDocument(
            query_id="q",
            document_rank=1,
            doc_id="d",
            source_item_rank=1,
            source_item_score=0.9,
            source_item_fused_score=0.9,
            source_item_semantic_score=0.8,
            source_item_metadata_score=0.5,
            membership_position=1,
        )


def test_expansion_version_is_exposed() -> None:
    assert EXPANSION_VERSION == "0.2.0"
