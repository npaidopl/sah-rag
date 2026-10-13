"""Tests for rag_system.baselines.metadata_rag_adapter and _state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag_system.baselines.metadata_rag_adapter import (
    ADAPTER_VERSION,
    INDEXED_DOCUMENT_VERSION,
    MetadataRAGProjectedQuery,
    build_chunk_id,
    build_indexed_document,
    build_indexed_documents,
    build_metadata_record_id,
    build_parent_id,
    build_window_id,
    project_queries,
)
from rag_system.baselines.metadata_rag_metadata import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    build_document_metadata,
)
from rag_system.baselines.metadata_rag_state import (
    STATE_VERSION,
    MetadataRAGState,
    build_state,
)
from rag_system.canonical.loaders import iter_validated_models
from rag_system.canonical.models import (
    CanonicalDocument,
    CanonicalExample,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = (
    REPO_ROOT
    / "processed"
    / "canonical"
    / "infrastructure-pilot-synthetic-v1"
)
FIXTURE_DATASET = "infrastructure-pilot-synthetic-v1"


class _FixedSizeSplitter:
    """Test-only splitter that splits text into fixed-size char chunks.

    Produces N windows of roughly equal character length. Satisfies the
    WindowSplitterProtocol. Not used in production.
    """

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars

    @property
    def max_content_tokens(self) -> int:
        return self._max_chars

    def split(self, text: str) -> tuple[Any, ...]:
        from rag_system.baselines.metadata_rag_windowing import WindowSpan

        if not text:
            raise ValueError("text must be non-empty")
        spans: list[WindowSpan] = []
        n = len(text)
        i = 0
        idx = 0
        while i < n:
            end = min(n, i + self._max_chars)
            spans.append(
                WindowSpan(
                    window_index=idx, char_start=i, char_end=end
                )
            )
            i = end
            idx += 1
        return tuple(spans)


def _load_fixture_documents() -> tuple[CanonicalDocument, ...]:
    return tuple(
        iter_validated_models(
            FIXTURE_ROOT / "documents.jsonl", CanonicalDocument
        )
    )


def _load_fixture_examples() -> tuple[CanonicalExample, ...]:
    return tuple(
        iter_validated_models(
            FIXTURE_ROOT / "examples.jsonl", CanonicalExample
        )
    )


def _sample_document() -> CanonicalDocument:
    return CanonicalDocument(
        doc_id="pilot-doc-01",
        dataset=FIXTURE_DATASET,
        text_raw="The Amber Observatory records data.",
        text_normalized="The Amber Observatory records data.",
        title=None,
        metadata={
            "source_dataset_id": FIXTURE_DATASET,
            "source_content_group_id": "pilot-content-group-01",
        },
    )


# --- ID derivation ---------------------------------------------------------


def test_build_parent_id_and_chunk_id_deterministic() -> None:
    assert build_parent_id("pilot-doc-01") == "pilot-doc-01:p0"
    assert build_chunk_id("pilot-doc-01") == "pilot-doc-01:c0"


def test_build_parent_id_rejects_empty() -> None:
    with pytest.raises(ValueError):
        build_parent_id("")


def test_build_chunk_id_with_window_index() -> None:
    assert build_chunk_id("d", 2) == "d:c2"


def test_build_chunk_id_rejects_negative_window_index() -> None:
    with pytest.raises(ValueError):
        build_chunk_id("d", -1)


def test_build_window_id_deterministic() -> None:
    assert build_window_id("d", 3) == "d:w3"


def test_build_window_id_rejects_negative() -> None:
    with pytest.raises(ValueError):
        build_window_id("d", -1)


def test_metadata_record_id_is_content_addressable() -> None:
    meta = build_document_metadata(
        dataset=FIXTURE_DATASET,
        doc_id="pilot-doc-01",
        source_dataset_id=FIXTURE_DATASET,
        text_normalized="The Amber Observatory records data.",
    )
    id_a = build_metadata_record_id(
        dataset_id=FIXTURE_DATASET, doc_id="pilot-doc-01", metadata=meta
    )
    id_b = build_metadata_record_id(
        dataset_id=FIXTURE_DATASET, doc_id="pilot-doc-01", metadata=meta
    )
    assert id_a == id_b
    assert id_a.startswith("mdrec-")


def test_metadata_record_id_changes_with_content() -> None:
    meta_a = build_document_metadata(
        dataset=FIXTURE_DATASET,
        doc_id="pilot-doc-01",
        source_dataset_id=FIXTURE_DATASET,
        text_normalized="The Amber Observatory records data.",
    )
    meta_b = build_document_metadata(
        dataset=FIXTURE_DATASET,
        doc_id="pilot-doc-01",
        source_dataset_id=FIXTURE_DATASET,
        text_normalized="The Cedar Archive stores documents.",
    )
    id_a = build_metadata_record_id(
        dataset_id=FIXTURE_DATASET, doc_id="pilot-doc-01", metadata=meta_a
    )
    id_b = build_metadata_record_id(
        dataset_id=FIXTURE_DATASET, doc_id="pilot-doc-01", metadata=meta_b
    )
    assert id_a != id_b


# --- Adapter: single-window (default) -------------------------------------


def test_build_indexed_document_shape() -> None:
    doc = _sample_document()
    indexed = build_indexed_document(
        document=doc, document_order=0, dataset_id=FIXTURE_DATASET
    )
    assert indexed.record_type == "metadata_rag_indexed_document"
    assert indexed.record_version == INDEXED_DOCUMENT_VERSION
    assert indexed.adapter_version == ADAPTER_VERSION
    assert indexed.schema_version == SCHEMA_VERSION
    assert indexed.extractor_version == EXTRACTOR_VERSION
    assert indexed.dataset_id == FIXTURE_DATASET
    assert indexed.lineage_doc_id == "pilot-doc-01"
    assert indexed.doc_id == "pilot-doc-01"
    assert indexed.window_index == 0
    assert indexed.parent_id == "pilot-doc-01:p0"
    assert indexed.chunk_id == "pilot-doc-01:c0"
    assert indexed.document_order == 0
    assert indexed.text_normalized == doc.text_normalized
    assert indexed.token_count == 5
    assert indexed.metadata_record_id.startswith("mdrec-")
    assert indexed.metadata.source_kind == "document"
    assert indexed.metadata.doc_id == "pilot-doc-01"
    assert indexed.metadata.capitalized_phrases == ("amber observatory",)


def test_build_indexed_document_rejects_negative_order() -> None:
    with pytest.raises(ValueError):
        build_indexed_document(
            document=_sample_document(),
            document_order=-1,
            dataset_id=FIXTURE_DATASET,
        )


def test_build_indexed_document_rejects_empty_dataset_id() -> None:
    with pytest.raises(ValueError):
        build_indexed_document(
            document=_sample_document(),
            document_order=0,
            dataset_id="",
        )


def test_build_indexed_document_rejects_window_index_out_of_range() -> None:
    with pytest.raises(ValueError):
        build_indexed_document(
            document=_sample_document(),
            document_order=0,
            dataset_id=FIXTURE_DATASET,
            window_index=5,
            n_windows=3,
            window_text="slice",
        )


def test_build_indexed_document_rejects_empty_window_text() -> None:
    with pytest.raises(ValueError, match="window_text"):
        build_indexed_document(
            document=_sample_document(),
            document_order=0,
            dataset_id=FIXTURE_DATASET,
            window_index=0,
            n_windows=1,
            window_text="   ",
        )


def test_build_indexed_documents_preserves_order_and_duplicates() -> None:
    docs = _load_fixture_documents()
    indexed = build_indexed_documents(docs, dataset_id=FIXTURE_DATASET)
    assert len(indexed) == 12
    for order, rec in enumerate(indexed):
        assert rec.document_order == order
        assert rec.window_index == 0
        assert rec.doc_id == rec.lineage_doc_id
    ids = [r.doc_id for r in indexed]
    assert ids[:2] == ["pilot-doc-01", "pilot-doc-02"]
    assert len(set(ids)) == 12


# --- Adapter: multi-window -------------------------------------------------


def test_build_indexed_documents_with_multi_window_splitter() -> None:
    doc = _sample_document()
    # 44-char text, split into ~10-char windows.
    splitter = _FixedSizeSplitter(max_chars=10)
    indexed = build_indexed_documents(
        (doc,), dataset_id=FIXTURE_DATASET, splitter=splitter
    )
    assert len(indexed) == 4  # ceil(35 / 10)
    for i, rec in enumerate(indexed):
        assert rec.window_index == i
        assert rec.lineage_doc_id == "pilot-doc-01"
        assert rec.doc_id == f"pilot-doc-01:w{i}"
        assert rec.chunk_id == f"pilot-doc-01:c{i}"
        assert rec.parent_id == "pilot-doc-01:p0"
        assert rec.document_order == i
    # Every window's text is a contiguous slice of the parent.
    joined = "".join(rec.text_normalized for rec in indexed)
    assert joined == doc.text_normalized


def test_multi_window_preserves_lineage_doc_id() -> None:
    doc = _sample_document()
    splitter = _FixedSizeSplitter(max_chars=10)
    indexed = build_indexed_documents(
        (doc,), dataset_id=FIXTURE_DATASET, splitter=splitter
    )
    lineage_ids = {rec.lineage_doc_id for rec in indexed}
    assert lineage_ids == {"pilot-doc-01"}


def test_multi_window_doc_ids_are_unique() -> None:
    doc = _sample_document()
    splitter = _FixedSizeSplitter(max_chars=10)
    indexed = build_indexed_documents(
        (doc,), dataset_id=FIXTURE_DATASET, splitter=splitter
    )
    doc_ids = [rec.doc_id for rec in indexed]
    assert len(set(doc_ids)) == len(doc_ids)


def test_multi_window_metadata_is_per_window() -> None:
    doc = _sample_document()
    # max_chars=25 keeps the whole two-token phrase "Amber Observatory"
    # inside the first window, because the phrase spans char positions
    # 4..21 of the 35-char sample text.
    splitter = _FixedSizeSplitter(max_chars=25)
    indexed = build_indexed_documents(
        (doc,), dataset_id=FIXTURE_DATASET, splitter=splitter
    )
    # The metadata record's doc_id is the canonical document, not the
    # window id, because the metadata tier is provenance.
    for rec in indexed:
        assert rec.metadata.doc_id == "pilot-doc-01"
    # The first window carries the full capitalized phrase.
    assert (
        "amber observatory"
        in indexed[0].metadata.capitalized_phrases
    )


def test_build_state_from_multi_window_index() -> None:
    doc = _sample_document()
    splitter = _FixedSizeSplitter(max_chars=10)
    indexed = build_indexed_documents(
        (doc,), dataset_id=FIXTURE_DATASET, splitter=splitter
    )
    state = build_state(indexed, dataset_id=FIXTURE_DATASET)
    assert state.state_version == STATE_VERSION
    assert len(state.documents) == len(indexed)


# --- Query projection ------------------------------------------------------


def test_project_queries_contains_only_id_and_text() -> None:
    examples = _load_fixture_examples()
    projected = project_queries(examples)
    assert len(projected) == 6
    for pq in projected:
        assert isinstance(pq, MetadataRAGProjectedQuery)
        assert set(pq.model_dump().keys()) == {"query_id", "query_text"}
        assert pq.query_id.startswith("pilot-query-")


def test_project_queries_preserves_query_text() -> None:
    examples = _load_fixture_examples()
    projected = project_queries(examples)
    assert projected[0].query_text == (
        "Which instruments record wind direction and air-pressure "
        "changes at the Amber Observatory?"
    )


# --- State -----------------------------------------------------------------


def test_build_state_from_fixture() -> None:
    docs = _load_fixture_documents()
    indexed = build_indexed_documents(docs, dataset_id=FIXTURE_DATASET)
    state = build_state(indexed, dataset_id=FIXTURE_DATASET)
    assert isinstance(state, MetadataRAGState)
    assert state.state_version == STATE_VERSION
    assert state.schema_version == SCHEMA_VERSION
    assert state.extractor_version == EXTRACTOR_VERSION
    assert len(state.documents) == 12


def test_build_state_rejects_empty() -> None:
    with pytest.raises(ValueError):
        build_state((), dataset_id=FIXTURE_DATASET)


def test_state_rejects_duplicate_doc_id() -> None:
    doc = build_indexed_document(
        document=_sample_document(),
        document_order=0,
        dataset_id=FIXTURE_DATASET,
    )
    dup = doc.model_copy(update={"document_order": 1})
    with pytest.raises(ValueError, match="duplicate doc_id"):
        MetadataRAGState(
            dataset_id=FIXTURE_DATASET, documents=(doc, dup)
        )


def test_state_rejects_non_contiguous_order() -> None:
    doc = build_indexed_document(
        document=_sample_document(),
        document_order=0,
        dataset_id=FIXTURE_DATASET,
    )
    bad = doc.model_copy(update={"document_order": 7})
    with pytest.raises(ValueError, match="contiguously"):
        MetadataRAGState(dataset_id=FIXTURE_DATASET, documents=(bad,))


def test_state_rejects_dataset_mismatch() -> None:
    doc = build_indexed_document(
        document=_sample_document(),
        document_order=0,
        dataset_id=FIXTURE_DATASET,
    )
    with pytest.raises(ValueError):
        MetadataRAGState(dataset_id="other-dataset", documents=(doc,))


def test_indexed_document_is_frozen() -> None:
    doc = build_indexed_document(
        document=_sample_document(),
        document_order=0,
        dataset_id=FIXTURE_DATASET,
    )
    with pytest.raises(ValidationError):
        doc.doc_id = "changed"


def test_state_is_frozen() -> None:
    doc = build_indexed_document(
        document=_sample_document(),
        document_order=0,
        dataset_id=FIXTURE_DATASET,
    )
    state = build_state((doc,), dataset_id=FIXTURE_DATASET)
    with pytest.raises(ValidationError):
        state.dataset_id = "changed"
