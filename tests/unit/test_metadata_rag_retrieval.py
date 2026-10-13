"""Tests for rag_system.baselines.metadata_rag_retrieval."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_system.baselines.metadata_rag_adapter import (
    MetadataRAGProjectedQuery,
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
    ReferenceDenseIndex,
    build_reference_dense_index,
)
from rag_system.baselines.metadata_rag_retrieval import (
    FROZEN_ALPHA_METADATA,
    FROZEN_ALPHA_SEMANTIC,
    FROZEN_MIN_MATCH_RATIO,
    QueryRetrievalResult,
    RetrievalConfig,
    retrieve_batch,
    retrieve_query,
)
from rag_system.baselines.metadata_rag_state import (
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
    text: str,
    document_order: int,
    lineage_doc_id: str | None = None,
    window_index: int = 0,
    dataset_id: str = "fixture",
) -> DenseIndexEntry:
    lineage = lineage_doc_id if lineage_doc_id is not None else doc_id
    metadata = build_document_metadata(
        dataset=dataset_id,
        doc_id=lineage,
        source_dataset_id=dataset_id,
        text_normalized=text,
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


# --- RetrievalConfig --------------------------------------------------------


def test_config_defaults_match_frozen_spec() -> None:
    cfg = RetrievalConfig()
    assert cfg.document_limit == 100
    assert math.isclose(cfg.alpha_metadata, FROZEN_ALPHA_METADATA)
    assert math.isclose(cfg.alpha_semantic, FROZEN_ALPHA_SEMANTIC)
    assert math.isclose(cfg.min_match_ratio, FROZEN_MIN_MATCH_RATIO)


def test_config_rejects_tuned_alpha_metadata() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(alpha_metadata=0.5, alpha_semantic=0.5)


def test_config_rejects_tuned_alpha_semantic() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(alpha_metadata=0.3, alpha_semantic=0.5)


def test_config_rejects_tuned_min_match_ratio() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(min_match_ratio=0.9)


def test_config_rejects_zero_document_limit() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(document_limit=0)


def test_config_is_frozen() -> None:
    cfg = RetrievalConfig()
    with pytest.raises(ValidationError):
        cfg.document_limit = 50


# --- Simple pipeline behavior ---------------------------------------------


def _two_doc_index_single_match() -> ReferenceDenseIndex:
    """Doc A has the target phrase, doc B does not."""
    e0 = _entry(
        doc_id="d-a",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        text="The Amber Observatory records data.",
        document_order=0,
    )
    e1 = _entry(
        doc_id="d-b",
        vector=_unit(FROZEN_DENSE_DIMENSION, 1),
        text="The Cedar Archive stores documents.",
        document_order=1,
    )
    return ReferenceDenseIndex((e0, e1))


def test_single_constraint_filters_non_matching_doc() -> None:
    index = _two_doc_index_single_match()
    cfg = RetrievalConfig()
    fake = _FakeDenseEncoder(_base_config())
    (qv,) = fake.encode(["Where is the Amber Observatory?"])

    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q-1",
            query_text="Where is the Amber Observatory?",
        ),
        index=index,
        query_vector=qv,
        config=cfg,
    )
    assert isinstance(result, QueryRetrievalResult)
    assert result.metadata.fallback_used is False
    assert result.metadata.constraint_count == 1
    assert result.metadata.candidate_count == 1
    doc_ids = [item.canonical_doc_id for item in result.ranked_items]
    assert doc_ids == ["d-a"]


def test_query_with_no_constraints_triggers_fallback() -> None:
    index = _two_doc_index_single_match()
    cfg = RetrievalConfig()
    fake = _FakeDenseEncoder(_base_config())
    (qv,) = fake.encode(["plain lowercase question"])
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q-1",
            query_text="plain lowercase question",
        ),
        index=index,
        query_vector=qv,
        config=cfg,
    )
    assert result.metadata.fallback_used is True
    assert result.metadata.constraint_count == 0
    assert result.metadata.candidate_count == 2
    for item in result.ranked_items:
        assert item.metadata.fallback_flags.metadata_match_none is True
        assert item.metadata.metadata_score == 0.0
    assert math.isclose(
        result.metadata.alpha_metadata, 0.0, abs_tol=1e-9
    )


def test_constraints_with_zero_doc_matches_triggers_fallback() -> None:
    index = _two_doc_index_single_match()
    cfg = RetrievalConfig()
    fake = _FakeDenseEncoder(_base_config())
    (qv,) = fake.encode(["Where is the Winter Navigation School?"])
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q-1",
            query_text="Where is the Winter Navigation School?",
        ),
        index=index,
        query_vector=qv,
        config=cfg,
    )
    assert result.metadata.fallback_used is True
    assert result.metadata.constraint_count == 1
    assert result.metadata.candidate_count == 2
    for item in result.ranked_items:
        assert item.metadata.fallback_flags.metadata_match_none is True


# --- Fusion math -----------------------------------------------------------


def _fusion_fixture() -> tuple[ReferenceDenseIndex, RetrievalConfig]:
    # Query text yields two constraints: cap:amber observatory and num:42.
    # doc-a: matches cap only     (metadata_score = 0.5)
    # doc-b: matches cap and 42   (metadata_score = 1.0)
    # doc-c: matches nothing      (metadata_score = 0.0)
    e0 = _entry(
        doc_id="doc-a",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        text="The Amber Observatory records data.",
        document_order=0,
    )
    e1 = _entry(
        doc_id="doc-b",
        vector=_unit(FROZEN_DENSE_DIMENSION, 1),
        text="The Amber Observatory recorded 42 readings.",
        document_order=1,
    )
    e2 = _entry(
        doc_id="doc-c",
        vector=_unit(FROZEN_DENSE_DIMENSION, 2),
        text="The Cedar Archive stores documents.",
        document_order=2,
    )
    return ReferenceDenseIndex((e0, e1, e2)), RetrievalConfig()


def test_fusion_uses_frozen_coefficients() -> None:
    index, cfg = _fusion_fixture()
    qv = _unit(FROZEN_DENSE_DIMENSION, 0)
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q-fusion",
            query_text="Amber Observatory 42",
        ),
        index=index,
        query_vector=qv,
        config=cfg,
    )
    assert result.metadata.constraint_count == 2
    by_doc = {i.canonical_doc_id: i for i in result.ranked_items}
    assert set(by_doc) == {"doc-a", "doc-b"}
    a = by_doc["doc-a"]
    b = by_doc["doc-b"]
    assert math.isclose(a.metadata.semantic_score, 1.0, abs_tol=1e-6)
    assert math.isclose(b.metadata.semantic_score, 0.0, abs_tol=1e-6)
    assert math.isclose(a.metadata.metadata_score, 0.5, abs_tol=1e-9)
    assert math.isclose(b.metadata.metadata_score, 1.0, abs_tol=1e-9)
    assert math.isclose(
        a.metadata.fused_score,
        0.3 * 0.5 + 0.7 * 1.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        b.metadata.fused_score,
        0.3 * 1.0 + 0.7 * 0.0,
        abs_tol=1e-6,
    )
    assert a.rank == 1
    assert b.rank == 2


def test_fusion_can_reorder_when_metadata_dominates() -> None:
    index, cfg = _fusion_fixture()
    qv = _unit(FROZEN_DENSE_DIMENSION, 1)
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q-fusion-b",
            query_text="Amber Observatory 42",
        ),
        index=index,
        query_vector=qv,
        config=cfg,
    )
    assert result.ranked_items[0].canonical_doc_id == "doc-b"
    assert result.ranked_items[1].canonical_doc_id == "doc-a"


def test_tie_breaks_by_semantic_score_descending() -> None:
    e0 = _entry(
        doc_id="aaa",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        text="The Amber Observatory records data.",
        document_order=0,
    )
    e1 = _entry(
        doc_id="bbb",
        vector=_unit(FROZEN_DENSE_DIMENSION, 1),
        text="The Amber Observatory records data.",
        document_order=1,
    )
    index = ReferenceDenseIndex((e0, e1))
    qv = _unit(FROZEN_DENSE_DIMENSION, 0)
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q",
            query_text="Amber Observatory",
        ),
        index=index,
        query_vector=qv,
        config=RetrievalConfig(),
    )
    assert result.ranked_items[0].canonical_doc_id == "aaa"
    assert result.ranked_items[1].canonical_doc_id == "bbb"


def test_tie_breaks_by_doc_id_when_both_scores_equal() -> None:
    e0 = _entry(
        doc_id="bbb",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        text="The Amber Observatory records data.",
        document_order=0,
    )
    e1 = _entry(
        doc_id="aaa",
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        text="The Amber Observatory records data.",
        document_order=1,
    )
    index = ReferenceDenseIndex((e0, e1))
    qv = _unit(FROZEN_DENSE_DIMENSION, 0)
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q",
            query_text="Amber Observatory",
        ),
        index=index,
        query_vector=qv,
        config=RetrievalConfig(),
    )
    assert [i.canonical_doc_id for i in result.ranked_items] == [
        "aaa",
        "bbb",
    ]


def test_ranked_item_ranks_are_contiguous_from_one() -> None:
    index, cfg = _fusion_fixture()
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q", query_text="Amber Observatory 42"
        ),
        index=index,
        query_vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        config=cfg,
    )
    assert [i.rank for i in result.ranked_items] == list(
        range(1, len(result.ranked_items) + 1)
    )


def test_document_limit_truncates_output() -> None:
    index, _ = _fusion_fixture()
    cfg = RetrievalConfig(document_limit=1)
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q", query_text="Amber Observatory 42"
        ),
        index=index,
        query_vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        config=cfg,
    )
    assert len(result.ranked_items) == 1


# --- Evidence and flags ----------------------------------------------------


def test_match_evidence_lists_satisfied_constraints() -> None:
    index, cfg = _fusion_fixture()
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q", query_text="Amber Observatory 42"
        ),
        index=index,
        query_vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        config=cfg,
    )
    by_doc = {i.canonical_doc_id: i for i in result.ranked_items}
    evidence_a = by_doc["doc-a"].metadata.metadata_match_evidence
    evidence_b = by_doc["doc-b"].metadata.metadata_match_evidence
    assert {e.constraint_id for e in evidence_a} == {
        "cap:amber observatory"
    }
    assert {e.constraint_id for e in evidence_b} == {
        "cap:amber observatory",
        "num:42",
    }


def test_metadata_missing_flag_set_when_doc_has_no_years() -> None:
    index, cfg = _fusion_fixture()
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q", query_text="Amber Observatory 42"
        ),
        index=index,
        query_vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        config=cfg,
    )
    by_doc = {i.canonical_doc_id: i for i in result.ranked_items}
    assert (
        by_doc["doc-a"].metadata.fallback_flags.metadata_missing_on_document
        is True
    )
    assert (
        by_doc["doc-b"].metadata.fallback_flags.metadata_missing_on_document
        is False
    )


def test_query_result_top_level_metadata_fields() -> None:
    index, cfg = _fusion_fixture()
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q", query_text="Amber Observatory 42"
        ),
        index=index,
        query_vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        config=cfg,
    )
    md = result.metadata
    assert math.isclose(md.alpha_metadata, 0.3, abs_tol=1e-9)
    assert math.isclose(md.alpha_semantic, 0.7, abs_tol=1e-9)
    assert md.fallback_used is False
    assert md.constraint_count == 2
    assert md.candidate_count == 2


def test_canonical_chunk_id_is_single_element() -> None:
    index, cfg = _fusion_fixture()
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q", query_text="Amber Observatory 42"
        ),
        index=index,
        query_vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        config=cfg,
    )
    for item in result.ranked_items:
        assert len(item.canonical_chunk_or_evidence_ids) == 1
        assert item.canonical_chunk_or_evidence_ids[0].startswith(
            item.canonical_doc_id
        )


# --- Multi-window behavior --------------------------------------------------


def test_multi_window_items_expose_lineage_as_canonical_doc_id() -> None:
    e0 = _entry(
        doc_id="d1:w0",
        lineage_doc_id="d1",
        window_index=0,
        vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        text="The Amber Observatory records data.",
        document_order=0,
    )
    e1 = _entry(
        doc_id="d1:w1",
        lineage_doc_id="d1",
        window_index=1,
        vector=_unit(FROZEN_DENSE_DIMENSION, 1),
        text="The Amber Observatory recorded more.",
        document_order=1,
    )
    index = ReferenceDenseIndex((e0, e1))
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id="q",
            query_text="Amber Observatory",
        ),
        index=index,
        query_vector=_unit(FROZEN_DENSE_DIMENSION, 0),
        config=RetrievalConfig(),
    )
    assert len(result.ranked_items) == 2
    for item in result.ranked_items:
        assert item.canonical_doc_id == "d1"
        assert item.architecture_item_id.startswith("d1:w")
        assert item.canonical_chunk_or_evidence_ids[0].startswith("d1:c")


# --- Input validation ------------------------------------------------------


def test_retrieve_query_rejects_non_projected_query() -> None:
    index, cfg = _fusion_fixture()
    with pytest.raises(TypeError):
        retrieve_query(
            query="not-a-query",  # type: ignore[arg-type]
            index=index,
            query_vector=_unit(FROZEN_DENSE_DIMENSION, 0),
            config=cfg,
        )


def test_retrieve_query_rejects_non_index() -> None:
    cfg = RetrievalConfig()
    with pytest.raises(TypeError):
        retrieve_query(
            query=MetadataRAGProjectedQuery(
                query_id="q", query_text="x"
            ),
            index="not-an-index",  # type: ignore[arg-type]
            query_vector=_unit(FROZEN_DENSE_DIMENSION, 0),
            config=cfg,
        )


# --- Batch -----------------------------------------------------------------


def test_retrieve_batch_preserves_order_and_shape() -> None:
    index, cfg = _fusion_fixture()
    queries = (
        MetadataRAGProjectedQuery(
            query_id="q-1", query_text="Amber Observatory"
        ),
        MetadataRAGProjectedQuery(
            query_id="q-2", query_text="Cedar Archive"
        ),
    )
    vectors = (
        _unit(FROZEN_DENSE_DIMENSION, 0),
        _unit(FROZEN_DENSE_DIMENSION, 2),
    )
    results = retrieve_batch(
        queries=queries,
        index=index,
        query_vectors=vectors,
        config=cfg,
    )
    assert len(results) == 2
    assert results[0].query_id == "q-1"
    assert results[1].query_id == "q-2"


def test_retrieve_batch_rejects_length_mismatch() -> None:
    index, cfg = _fusion_fixture()
    queries = (
        MetadataRAGProjectedQuery(query_id="q-1", query_text="x"),
    )
    with pytest.raises(Exception, match="identical length"):
        retrieve_batch(
            queries=queries,
            index=index,
            query_vectors=(
                _unit(FROZEN_DENSE_DIMENSION, 0),
                _unit(FROZEN_DENSE_DIMENSION, 1),
            ),
            config=cfg,
        )


# --- Fixture integration ---------------------------------------------------


def _fixture_index() -> ReferenceDenseIndex:
    docs = tuple(
        iter_validated_models(
            FIXTURE_ROOT / "documents.jsonl", CanonicalDocument
        )
    )
    indexed = build_indexed_documents(docs, dataset_id=FIXTURE_DATASET)
    state: MetadataRAGState = build_state(
        indexed, dataset_id=FIXTURE_DATASET
    )
    cfg = _base_config()
    fake = _FakeDenseEncoder(cfg)
    vectors = fake.encode([d.text_normalized for d in state.documents])
    embeddings: tuple[MetadataRAGDocumentEmbedding, ...] = tuple(
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
    return build_reference_dense_index(
        state=state, embeddings=embeddings
    )


def test_fixture_pipeline_returns_ground_truth_documents() -> None:
    """Every fixture query must return its ground-truth docs at top-2.

    The synthetic fixture is intentionally designed so that the single
    capitalized-phrase constraint in each query isolates exactly the
    two ground-truth documents. This test pins that property end to
    end through the full retrieval pipeline.
    """
    index = _fixture_index()
    cfg = RetrievalConfig(document_limit=12)
    fake = _FakeDenseEncoder(_base_config())

    examples = tuple(
        iter_validated_models(
            FIXTURE_ROOT / "examples.jsonl", CanonicalExample
        )
    )
    assert len(examples) == 6

    for example in examples:
        query = MetadataRAGProjectedQuery(
            query_id=example.query.query_id,
            query_text=example.query.question,
        )
        (qv,) = fake.encode([query.query_text])
        result = retrieve_query(
            query=query,
            index=index,
            query_vector=qv,
            config=cfg,
        )
        assert result.metadata.fallback_used is False
        assert result.metadata.constraint_count == 1
        top_two = [
            i.canonical_doc_id for i in result.ranked_items[:2]
        ]
        assert top_two == list(example.ground_truth.relevant_ids), (
            f"{example.query.query_id}: got {top_two}"
        )


def test_fixture_lineage_equals_doc_id_for_single_window() -> None:
    index = _fixture_index()
    cfg = RetrievalConfig(document_limit=12)
    fake = _FakeDenseEncoder(_base_config())

    examples = tuple(
        iter_validated_models(
            FIXTURE_ROOT / "examples.jsonl", CanonicalExample
        )
    )
    (qv,) = fake.encode([examples[0].query.question])
    result = retrieve_query(
        query=MetadataRAGProjectedQuery(
            query_id=examples[0].query.query_id,
            query_text=examples[0].query.question,
        ),
        index=index,
        query_vector=qv,
        config=cfg,
    )
    for item in result.ranked_items:
        # Fixture is single-window: architecture_item_id == canonical.
        assert item.architecture_item_id == item.canonical_doc_id
        assert item.canonical_chunk_or_evidence_ids == (
            build_chunk_id(item.canonical_doc_id, 0),
        )
