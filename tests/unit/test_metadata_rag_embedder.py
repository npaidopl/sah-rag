"""Tests for rag_system.baselines.metadata_rag_embedder."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from rag_system.baselines.metadata_rag_embedder import (
    EMBEDDER_VERSION,
    FROZEN_DENSE_DIMENSION,
    FROZEN_MODEL_ID,
    FROZEN_MODEL_REVISION,
    BgeM3DenseEncoder,
    DenseEmbedderProtocol,
    EncoderConfig,
    MetadataRAGDocumentEmbedding,
    MetadataRAGEmbedderError,
    build_document_embedding,
    build_query_embedding,
)

# --- A deterministic fake encoder used only in tests ----------------------


class _FakeDenseEncoder:
    """Deterministic encoder that produces unit vectors from text hashes.

    The vector is derived from the SHA-256 digest of the text,
    repeated and truncated to 1024 bytes, mapped to floats, then
    L2-normalized. Deterministic and offline.
    """

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


def _base_config() -> EncoderConfig:
    return EncoderConfig(local_snapshot_path="C:/fake/snapshot")


# --- EncoderConfig ---------------------------------------------------------


def test_config_defaults_match_frozen_spec() -> None:
    cfg = _base_config()
    assert cfg.model_id == FROZEN_MODEL_ID
    assert cfg.model_revision == FROZEN_MODEL_REVISION
    assert cfg.dense_dimension == FROZEN_DENSE_DIMENSION
    assert cfg.dense_normalized is True
    assert cfg.network_fallback_allowed is False
    assert cfg.device == "cpu"
    assert cfg.use_fp16 is False


def test_config_rejects_wrong_model_id() -> None:
    with pytest.raises(ValidationError):
        EncoderConfig(model_id="not-bge-m3", local_snapshot_path="x")


def test_config_rejects_wrong_revision() -> None:
    with pytest.raises(ValidationError):
        EncoderConfig(model_revision="deadbeef", local_snapshot_path="x")


def test_config_rejects_unnormalized_vectors() -> None:
    with pytest.raises(ValidationError):
        EncoderConfig(dense_normalized=False, local_snapshot_path="x")


def test_config_rejects_network_fallback() -> None:
    with pytest.raises(ValidationError):
        EncoderConfig(
            network_fallback_allowed=True, local_snapshot_path="x"
        )


def test_config_hash_is_stable_and_content_addressed() -> None:
    a = EncoderConfig(local_snapshot_path="C:/fake/snapshot")
    b = EncoderConfig(local_snapshot_path="C:/fake/snapshot")
    c = EncoderConfig(local_snapshot_path="C:/other/snapshot")
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()
    assert len(a.config_hash()) == 64


def test_config_is_frozen() -> None:
    cfg = _base_config()
    with pytest.raises(ValidationError):
        cfg.device = "cuda"


# --- Protocol compliance ---------------------------------------------------


def test_fake_encoder_satisfies_protocol() -> None:
    fake = _FakeDenseEncoder(_base_config())
    assert isinstance(fake, DenseEmbedderProtocol)


def test_bge_m3_encoder_satisfies_protocol() -> None:
    enc = BgeM3DenseEncoder(_base_config())
    assert isinstance(enc, DenseEmbedderProtocol)


def test_bge_m3_encoder_does_not_load_model_at_construction() -> None:
    # If the model were loaded here, this test would fail because the
    # snapshot path is fake.
    enc = BgeM3DenseEncoder(_base_config())
    assert enc.config.model_id == FROZEN_MODEL_ID


def test_bge_m3_encoder_rejects_empty_snapshot() -> None:
    enc = BgeM3DenseEncoder(EncoderConfig())
    with pytest.raises(
        MetadataRAGEmbedderError, match="local_snapshot_path"
    ):
        enc.encode(["hello"])


def test_bge_m3_encoder_rejects_empty_text() -> None:
    enc = BgeM3DenseEncoder(_base_config())
    with pytest.raises(MetadataRAGEmbedderError, match="non-empty"):
        enc.encode([""])


def test_bge_m3_encoder_returns_empty_for_empty_input() -> None:
    enc = BgeM3DenseEncoder(_base_config())
    assert enc.encode(()) == ()


def test_fake_encoder_vectors_are_normalized() -> None:
    fake = _FakeDenseEncoder(_base_config())
    vectors = fake.encode(["alpha", "beta", "gamma"])
    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == FROZEN_DENSE_DIMENSION
        norm = math.sqrt(sum(v * v for v in vector))
        assert math.isclose(norm, 1.0, abs_tol=1e-9)


# --- Embedding records -----------------------------------------------------


def test_build_document_embedding_shape() -> None:
    cfg = _base_config()
    fake = _FakeDenseEncoder(cfg)
    (vector,) = fake.encode(["The Amber Observatory records data."])
    rec = build_document_embedding(
        config=cfg,
        dataset_id="infrastructure-pilot-synthetic-v1",
        doc_id="pilot-doc-01",
        document_order=0,
        metadata_record_id="mdrec-deadbeef",
        source_text="The Amber Observatory records data.",
        dense_vector=vector,
    )
    assert rec.record_kind == "document_embedding"
    assert rec.embedding_version == EMBEDDER_VERSION
    assert rec.model_id == FROZEN_MODEL_ID
    assert rec.model_revision == FROZEN_MODEL_REVISION
    assert rec.embedding_config_hash == cfg.config_hash()
    assert rec.dense_dimension == FROZEN_DENSE_DIMENSION
    assert rec.dense_normalized is True
    assert len(rec.source_text_sha256) == 64


def test_build_query_embedding_shape() -> None:
    cfg = _base_config()
    fake = _FakeDenseEncoder(cfg)
    (vector,) = fake.encode(["Which instrument records wind direction?"])
    rec = build_query_embedding(
        config=cfg,
        dataset_id="infrastructure-pilot-synthetic-v1",
        query_id="pilot-query-01",
        query_text="Which instrument records wind direction?",
        dense_vector=vector,
    )
    assert rec.record_kind == "query_embedding"
    assert rec.query_id == "pilot-query-01"
    assert rec.dense_dimension == FROZEN_DENSE_DIMENSION
    assert len(rec.query_text_sha256) == 64


def test_query_embedding_does_not_store_query_text() -> None:
    cfg = _base_config()
    fake = _FakeDenseEncoder(cfg)
    text = "Which instrument records wind direction?"
    (vector,) = fake.encode([text])
    rec = build_query_embedding(
        config=cfg,
        dataset_id="ds",
        query_id="q-1",
        query_text=text,
        dense_vector=vector,
    )
    dumped = rec.model_dump()
    assert "query_text" not in dumped
    assert "query_text_sha256" in dumped
    assert text not in str(dumped)


def test_build_document_embedding_rejects_wrong_dimension() -> None:
    cfg = _base_config()
    with pytest.raises(MetadataRAGEmbedderError, match="length"):
        build_document_embedding(
            config=cfg,
            dataset_id="ds",
            doc_id="d",
            document_order=0,
            metadata_record_id="mdrec-x",
            source_text="text",
            dense_vector=(0.5, 0.5),
        )


def test_build_document_embedding_rejects_unnormalized_vector() -> None:
    cfg = _base_config()
    vector = tuple([2.0] * FROZEN_DENSE_DIMENSION)
    with pytest.raises(MetadataRAGEmbedderError, match="normalized"):
        build_document_embedding(
            config=cfg,
            dataset_id="ds",
            doc_id="d",
            document_order=0,
            metadata_record_id="mdrec-x",
            source_text="text",
            dense_vector=vector,
        )


def test_build_document_embedding_rejects_nan() -> None:
    cfg = _base_config()
    vector = tuple(
        [float("nan")] + [0.0] * (FROZEN_DENSE_DIMENSION - 1)
    )
    with pytest.raises(MetadataRAGEmbedderError, match="non-finite"):
        build_document_embedding(
            config=cfg,
            dataset_id="ds",
            doc_id="d",
            document_order=0,
            metadata_record_id="mdrec-x",
            source_text="text",
            dense_vector=vector,
        )


def test_build_document_embedding_rejects_empty_source_text() -> None:
    cfg = _base_config()
    fake = _FakeDenseEncoder(cfg)
    (vector,) = fake.encode(["something"])
    with pytest.raises(MetadataRAGEmbedderError, match="source_text"):
        build_document_embedding(
            config=cfg,
            dataset_id="ds",
            doc_id="d",
            document_order=0,
            metadata_record_id="mdrec-x",
            source_text="   ",
            dense_vector=vector,
        )


def test_embedding_records_are_frozen() -> None:
    cfg = _base_config()
    fake = _FakeDenseEncoder(cfg)
    (vector,) = fake.encode(["text"])
    rec = build_document_embedding(
        config=cfg,
        dataset_id="ds",
        doc_id="d",
        document_order=0,
        metadata_record_id="mdrec-x",
        source_text="text",
        dense_vector=vector,
    )
    with pytest.raises(ValidationError):
        rec.doc_id = "changed"


def test_embedding_record_rejects_wrong_model_id_at_construction() -> None:
    with pytest.raises(ValidationError):
        MetadataRAGDocumentEmbedding(
            metadata_record_id="mdrec-x",
            dataset_id="ds",
            doc_id="d",
            document_order=0,
            source_text_sha256="0" * 64,
            model_id="not-bge-m3",
            model_revision=FROZEN_MODEL_REVISION,
            embedding_config_hash="h",
            dense_vector=tuple([0.0] * FROZEN_DENSE_DIMENSION),
            dense_dimension=FROZEN_DENSE_DIMENSION,
            dense_normalized=True,
        )
