"""Tests for rag_system.baselines.metadata_rag_reference_index."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_system.baselines.metadata_rag_adapter import (
    build_chunk_id,
    build_indexed_documents,
)
from rag_system.baselines.metadata_rag_embedder import (
    FROZEN_DENSE_DIMENSION,
    EncoderConfig,
    MetadataRAGDocumentEmbedding,
    build_document_embedding,
)
from rag_system.baselines.metadata_rag_metadata import (
    build_document_metadata,
)
from rag_system.baselines.metadata_rag_reference_index import (
    DenseIndexEntry,
    MetadataRAGReferenceIndexError,
    ReferenceDenseIndex,
    build_reference_dense_index,
)
from rag_system.baselines.metadata_rag_state import (
    MetadataRAGState,
    build_state,
)
from rag_system.canonical.loaders import iter_validated_models
from rag_system.canonical.models import CanonicalDocument

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = (
    REPO_ROOT
    / "processed"
    / "canonical"
    / "infrastructure-pilot-synthetic-v1"
)
FIXTURE_DATASET = "infrastructure-pilot-synthetic-v1"


class _FakeDenseEncoder:
    def __init__(self, config: EncoderConfig) -> None:
        self._config = config

    @property
    def config(self) -> EncoderConfig:
        return self._config

    def encode(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        rows: list[tuple[float, ...]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = (digest * (FROZEN_DENSE_DIMENSION // len(digest) + 1))[
                :FROZEN_DENSE_DIMENSION
            ]
            vec = tuple(float(b) - 128.0 for b in raw)
            norm = math.sqrt(sum(v * v for v in vec))
            rows.append(tuple(v / norm for v in vec))
        return tuple(rows)


def _unit(dim: int, index: int) -> tuple[float, ...]:
    vec = [0.0] * dim
    vec[index] = 1.0
    return tuple(vec)


def _base_config() -> EncoderConfig:
    return EncoderConfig(local_snapshot_path="C:/fake/snapshot")


def _entry(
    *,
    doc_id: str,
    vector: tuple[float, ...],
    lineage_doc_id: str | None = None,
    window_index: int = 0,
    dataset_id: str = "fixture",
    document_order: int = 0,
) -> DenseIndexEntry:
    lineage = lineage_doc_id if lineage_doc_id is not None else doc_id
    metadata = build_document_metadata(
        dataset=dataset_id,
        doc_id=lineage,
        source_dataset_id=dataset_id,
        text_normalized="placeholder text",
    )
    return DenseIndexEntry(
        dataset_id=dataset_id,
        lineage_doc_id=lineage,
        doc_id=doc_id,
        window_index=window_index,
        metadata_record_id=f"mdrec-{doc_id}",
        document_order=document_order,
        dense_vector=vector,
        metadata=metadata,
    )


# --- Entry model ----------------------------------------------------------


def test_entry_accepts_single_window_shape() -> None:
    entry = _entry(doc_id="d1", vector=_unit(FROZEN_DENSE_DIMENSION, 0))
    assert entry.doc_id == "d1"
    assert entry.lineage_doc_id == "d1"
    assert entry.window_index == 0
    assert entry.metadata.source_kind == "document"


def test_entry_accepts_multi_window_shape() -> None:
    entry = _entry(
        doc_id="d1:w3",
        lineage_doc_id="d1",
        window_index=3,
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
    )
    assert entry.doc_id == "d1:w3"
    assert entry.lineage_doc_id == "d1"
    assert entry.window_index == 3
    assert entry.metadata.doc_id == "d1"


def test_entry_rejects_wrong_dimension() -> None:
    with pytest.raises(ValidationError):
        _entry(doc_id="d1", vector=(0.5, 0.5))


def test_entry_rejects_unnormalized_vector() -> None:
    bad = tuple([2.0] + [0.0] * (FROZEN_DENSE_DIMENSION - 1))
    with pytest.raises(ValidationError, match="normalized"):
        _entry(doc_id="d1", vector=bad)


def test_entry_rejects_nan() -> None:
    bad = tuple(
        [float("nan")] + [0.0] * (FROZEN_DENSE_DIMENSION - 1)
    )
    with pytest.raises(ValidationError, match="non-finite"):
        _entry(doc_id="d1", vector=bad)


def test_entry_rejects_metadata_lineage_mismatch() -> None:
    metadata = build_document_metadata(
        dataset="fixture",
        doc_id="OTHER",
        source_dataset_id="fixture",
        text_normalized="text",
    )
    with pytest.raises(ValidationError, match="lineage_doc_id"):
        DenseIndexEntry(
            dataset_id="fixture",
            lineage_doc_id="d1",
            doc_id="d1",
            window_index=0,
            metadata_record_id="mdrec-d1",
            document_order=0,
            dense_vector=_unit(FROZEN_DENSE_DIMENSION, 0),
            metadata=metadata,
        )


# --- Index construction ---------------------------------------------------


def test_index_rejects_empty_entries() -> None:
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="at least one"
    ):
        ReferenceDenseIndex(())


def test_index_rejects_duplicate_doc_ids() -> None:
    e1 = _entry(
        doc_id="d1",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=0,
    )
    e2 = _entry(
        doc_id="d1",
        vector=_unit(FROZEN_DENSE_DIMENSION, 1),
        document_order=1,
    )
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="duplicate doc_id"
    ):
        ReferenceDenseIndex((e1, e2))


def test_index_rejects_non_contiguous_order() -> None:
    e1 = _entry(
        doc_id="d1",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=7,
    )
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="contiguously"
    ):
        ReferenceDenseIndex((e1,))


def test_index_rejects_mixed_datasets() -> None:
    e1 = _entry(
        doc_id="d1",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        dataset_id="ds-a",
        document_order=0,
    )
    e2 = _entry(
        doc_id="d2",
        vector=_unit(FROZEN_DENSE_DIMENSION, 1),
        dataset_id="ds-b",
        document_order=1,
    )
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="dataset_id"
    ):
        ReferenceDenseIndex((e1, e2))


def test_index_properties() -> None:
    e1 = _entry(
        doc_id="d1",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=0,
    )
    e2 = _entry(
        doc_id="d2",
        vector=_unit(FROZEN_DENSE_DIMENSION, 1),
        document_order=1,
    )
    index = ReferenceDenseIndex((e1, e2))
    assert index.size == 2
    assert index.dataset_id == "fixture"
    assert index.doc_ids == ("d1", "d2")
    assert index.dense_dimension == FROZEN_DENSE_DIMENSION


def test_index_accepts_multi_window_doc_ids() -> None:
    e0 = _entry(
        doc_id="d1:w0",
        lineage_doc_id="d1",
        window_index=0,
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=0,
    )
    e1 = _entry(
        doc_id="d1:w1",
        lineage_doc_id="d1",
        window_index=1,
        vector=_unit(FROZEN_DENSE_DIMENSION, 1),
        document_order=1,
    )
    index = ReferenceDenseIndex((e0, e1))
    assert index.doc_ids == ("d1:w0", "d1:w1")


def test_metadata_for_round_trip() -> None:
    e1 = _entry(
        doc_id="d1",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=0,
    )
    index = ReferenceDenseIndex((e1,))
    assert index.metadata_for("d1") == e1.metadata


def test_metadata_for_unknown_doc_id() -> None:
    e1 = _entry(
        doc_id="d1",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=0,
    )
    index = ReferenceDenseIndex((e1,))
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="unknown doc_id"
    ):
        index.metadata_for("missing")


def test_lineage_for_round_trip() -> None:
    e0 = _entry(
        doc_id="d1:w0",
        lineage_doc_id="d1",
        window_index=0,
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=0,
    )
    e1 = _entry(
        doc_id="d1:w1",
        lineage_doc_id="d1",
        window_index=1,
        vector=_unit(FROZEN_DENSE_DIMENSION, 1),
        document_order=1,
    )
    index = ReferenceDenseIndex((e0, e1))
    assert index.lineage_for("d1:w0") == "d1"
    assert index.lineage_for("d1:w1") == "d1"


def test_lineage_for_unknown_doc_id() -> None:
    e1 = _entry(
        doc_id="d1",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=0,
    )
    index = ReferenceDenseIndex((e1,))
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="unknown doc_id"
    ):
        index.lineage_for("missing")


# --- Search ---------------------------------------------------------------


def _tiny_index() -> ReferenceDenseIndex:
    e0 = _entry(
        doc_id="d-a",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=0,
    )
    e1 = _entry(
        doc_id="d-b",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=1,
    )
    e2 = _entry(
        doc_id="d-c",
        vector=_unit(FROZEN_DENSE_DIMENSION, 1),
        document_order=2,
    )
    return ReferenceDenseIndex((e0, e1, e2))


def test_search_top1_score_one_for_identical_vector() -> None:
    index = _tiny_index()
    hits = index.search(_unit(FROZEN_DENSE_DIMENSION, 0), top_k=1)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[0].rank == 1


def test_search_top_k_greater_than_size_returns_all() -> None:
    index = _tiny_index()
    hits = index.search(_unit(FROZEN_DENSE_DIMENSION, 0), top_k=100)
    assert len(hits) == 3


def test_search_top_k_zero_raises() -> None:
    index = _tiny_index()
    with pytest.raises(MetadataRAGReferenceIndexError, match="top_k"):
        index.search(_unit(FROZEN_DENSE_DIMENSION, 0), top_k=0)


def test_search_rejects_wrong_query_dimension() -> None:
    index = _tiny_index()
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="query_vector"
    ):
        index.search((1.0, 0.0), top_k=1)


def test_search_rejects_unnormalized_query() -> None:
    index = _tiny_index()
    bad = tuple([2.0] + [0.0] * (FROZEN_DENSE_DIMENSION - 1))
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="normalized"
    ):
        index.search(bad, top_k=1)


def test_search_rejects_non_finite_query() -> None:
    index = _tiny_index()
    bad = tuple(
        [float("inf")] + [0.0] * (FROZEN_DENSE_DIMENSION - 1)
    )
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="non-finite"
    ):
        index.search(bad, top_k=1)


def test_search_ties_break_by_doc_id_ascending() -> None:
    index = _tiny_index()
    hits = index.search(_unit(FROZEN_DENSE_DIMENSION, 0), top_k=2)
    assert [h.doc_id for h in hits] == ["d-a", "d-b"]
    assert all(
        h.score == pytest.approx(1.0, abs=1e-6) for h in hits
    )


def test_search_ranks_are_contiguous_from_one() -> None:
    index = _tiny_index()
    hits = index.search(_unit(FROZEN_DENSE_DIMENSION, 0), top_k=3)
    assert [h.rank for h in hits] == [1, 2, 3]


def test_search_with_candidate_filter() -> None:
    index = _tiny_index()
    hits = index.search(
        _unit(FROZEN_DENSE_DIMENSION, 0),
        top_k=10,
        candidate_doc_ids=frozenset({"d-c"}),
    )
    assert len(hits) == 1
    assert hits[0].doc_id == "d-c"


def test_search_with_empty_candidate_filter_returns_empty() -> None:
    index = _tiny_index()
    hits = index.search(
        _unit(FROZEN_DENSE_DIMENSION, 0),
        top_k=10,
        candidate_doc_ids=frozenset(),
    )
    assert hits == ()


def test_search_with_unknown_candidate_raises() -> None:
    index = _tiny_index()
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="not present in index"
    ):
        index.search(
            _unit(FROZEN_DENSE_DIMENSION, 0),
            top_k=10,
            candidate_doc_ids=frozenset({"nope"}),
        )


def test_search_is_deterministic() -> None:
    index = _tiny_index()
    q = _unit(FROZEN_DENSE_DIMENSION, 0)
    a = index.search(q, top_k=3)
    b = index.search(q, top_k=3)
    assert a == b


def test_search_scores_are_finite_and_bounded() -> None:
    index = _tiny_index()
    hits = index.search(_unit(FROZEN_DENSE_DIMENSION, 0), top_k=3)
    for h in hits:
        assert math.isfinite(h.score)
        assert -1.0001 <= h.score <= 1.0001


def test_search_hit_has_higher_is_better_semantics() -> None:
    index = _tiny_index()
    hits = index.search(_unit(FROZEN_DENSE_DIMENSION, 0), top_k=1)
    assert hits[0].score_semantics == "HIGHER_IS_BETTER"


def test_search_hit_carries_metadata_record_id() -> None:
    index = _tiny_index()
    hits = index.search(_unit(FROZEN_DENSE_DIMENSION, 0), top_k=1)
    assert hits[0].metadata_record_id == "mdrec-d-a"


def test_search_hit_carries_lineage_and_chunk_id_for_single() -> None:
    index = _tiny_index()
    hits = index.search(_unit(FROZEN_DENSE_DIMENSION, 0), top_k=1)
    assert hits[0].lineage_doc_id == "d-a"
    assert hits[0].window_index == 0
    assert hits[0].chunk_id == "d-a:c0"


def test_search_hit_carries_lineage_and_chunk_id_for_windows() -> None:
    e0 = _entry(
        doc_id="d1:w0",
        lineage_doc_id="d1",
        window_index=0,
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=0,
    )
    e1 = _entry(
        doc_id="d1:w1",
        lineage_doc_id="d1",
        window_index=1,
        vector=_unit(FROZEN_DENSE_DIMENSION, 1),
        document_order=1,
    )
    index = ReferenceDenseIndex((e0, e1))
    hits = index.search(_unit(FROZEN_DENSE_DIMENSION, 0), top_k=2)
    assert hits[0].doc_id == "d1:w0"
    assert hits[0].lineage_doc_id == "d1"
    assert hits[0].window_index == 0
    assert hits[0].chunk_id == "d1:c0"
    assert hits[1].doc_id == "d1:w1"
    assert hits[1].lineage_doc_id == "d1"
    assert hits[1].window_index == 1
    assert hits[1].chunk_id == "d1:c1"


def test_search_windows_of_one_doc_tie_break_by_window_id() -> None:
    e0 = _entry(
        doc_id="d1:w0",
        lineage_doc_id="d1",
        window_index=0,
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=0,
    )
    e1 = _entry(
        doc_id="d1:w1",
        lineage_doc_id="d1",
        window_index=1,
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        document_order=1,
    )
    index = ReferenceDenseIndex((e0, e1))
    hits = index.search(_unit(FROZEN_DENSE_DIMENSION, 0), top_k=2)
    assert [h.doc_id for h in hits] == ["d1:w0", "d1:w1"]


# --- Builder from fixture --------------------------------------------------


def _fixture_state_and_embeddings() -> tuple[
    MetadataRAGState, tuple[MetadataRAGDocumentEmbedding, ...]
]:
    docs = tuple(
        iter_validated_models(
            FIXTURE_ROOT / "documents.jsonl", CanonicalDocument
        )
    )
    indexed = build_indexed_documents(docs, dataset_id=FIXTURE_DATASET)
    state = build_state(indexed, dataset_id=FIXTURE_DATASET)
    cfg = _base_config()
    fake = _FakeDenseEncoder(cfg)
    texts = [d.text_normalized for d in state.documents]
    vectors = fake.encode(texts)
    embeddings = tuple(
        build_document_embedding(
            config=cfg,
            dataset_id=state.dataset_id,
            doc_id=d.doc_id,
            document_order=d.document_order,
            metadata_record_id=d.metadata_record_id,
            source_text=d.text_normalized,
            dense_vector=v,
        )
        for d, v in zip(state.documents, vectors, strict=True)
    )
    return state, embeddings


def test_build_reference_index_from_fixture() -> None:
    state, embeddings = _fixture_state_and_embeddings()
    index = build_reference_dense_index(
        state=state, embeddings=embeddings
    )
    assert index.size == 12
    assert index.dataset_id == FIXTURE_DATASET


def test_builder_rejects_missing_embedding() -> None:
    state, embeddings = _fixture_state_and_embeddings()
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="missing embedding"
    ):
        build_reference_dense_index(
            state=state, embeddings=embeddings[:-1]
        )


def test_builder_rejects_extra_embedding() -> None:
    state, embeddings = _fixture_state_and_embeddings()
    dup = embeddings[0].model_copy(update={"doc_id": "pilot-doc-99"})
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="without matching"
    ):
        build_reference_dense_index(
            state=state, embeddings=(*embeddings, dup)
        )


def test_builder_rejects_metadata_record_id_mismatch() -> None:
    state, embeddings = _fixture_state_and_embeddings()
    bad = embeddings[0].model_copy(
        update={"metadata_record_id": "mdrec-WRONG"}
    )
    with pytest.raises(
        MetadataRAGReferenceIndexError,
        match="metadata_record_id mismatch",
    ):
        build_reference_dense_index(
            state=state, embeddings=(bad, *embeddings[1:])
        )


def test_builder_rejects_document_order_mismatch() -> None:
    state, embeddings = _fixture_state_and_embeddings()
    bad = embeddings[0].model_copy(update={"document_order": 7})
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="document_order mismatch"
    ):
        build_reference_dense_index(
            state=state, embeddings=(bad, *embeddings[1:])
        )


def test_builder_rejects_dataset_id_mismatch() -> None:
    state, embeddings = _fixture_state_and_embeddings()
    bad = embeddings[0].model_copy(update={"dataset_id": "other"})
    with pytest.raises(
        MetadataRAGReferenceIndexError, match="dataset_id"
    ):
        build_reference_dense_index(
            state=state, embeddings=(bad, *embeddings[1:])
        )


def test_search_on_fixture_returns_hits_with_canonical_doc_ids() -> None:
    state, embeddings = _fixture_state_and_embeddings()
    index = build_reference_dense_index(
        state=state, embeddings=embeddings
    )
    cfg = _base_config()
    fake = _FakeDenseEncoder(cfg)
    (query_vector,) = fake.encode(
        ["Which instruments record patterns at the Amber Observatory?"]
    )
    hits = index.search(query_vector, top_k=5)
    assert len(hits) == 5
    fixture_doc_ids = {d.doc_id for d in state.documents}
    for hit in hits:
        assert hit.doc_id in fixture_doc_ids
        # Single-window fixture: lineage equals doc_id.
        assert hit.lineage_doc_id == hit.doc_id
        assert hit.chunk_id == build_chunk_id(hit.lineage_doc_id, 0)
