"""Frozen BGE-M3 dense encoder for the Metadata-RAG comparator.

Frozen specification:
    baseline_audit/metadata_rag_clean_room_spec.yaml
Frozen amendments:
    baseline_audit/metadata_rag_clean_room_spec_amendment_01.yaml
    baseline_audit/metadata_rag_clean_room_spec_amendment_02.yaml
    baseline_audit/metadata_rag_clean_room_spec_amendment_03.yaml

The embedder is classified ARCHITECTURE_SUPPORTING with treatment
CONTROLLED_REPLACEMENT. Its identity is pinned by (model_id,
model_revision). Native equivalence with any hypothetical paper
embedder is not claimed. The encoder is dense-only, L2-normalized,
and network-free (a local snapshot is required).

Design notes:
  * The model is never loaded at import time. Loading is lazy, so
    unit tests and validators can run without the model weights.
  * FlagEmbedding is imported dynamically as an untyped module, so
    strict mypy does not require third-party stubs.
  * The encoder is stateless with respect to the caller: it does not
    cache, does not persist, and does not read ground truth.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

EMBEDDER_VERSION = "0.1.0"

FROZEN_MODEL_ID = "BAAI/bge-m3"
FROZEN_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FROZEN_DENSE_DIMENSION = 1024

_NORMALIZATION_TOLERANCE = 1e-3


class MetadataRAGEmbedderError(ValueError):
    """Raised when the encoder violates its frozen contract."""


class EncoderConfig(BaseModel):
    """Frozen encoder configuration.

    The configuration is content-addressable via ``config_hash``.
    Two configs that differ in any field produce different hashes, and
    the hash is part of every embedding record so that records cannot
    be silently mixed across encoder identities.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    model_id: str = FROZEN_MODEL_ID
    model_revision: str = FROZEN_MODEL_REVISION
    dense_dimension: int = FROZEN_DENSE_DIMENSION
    dense_normalized: bool = True
    device: Literal["cpu", "cuda"] = "cpu"
    use_fp16: bool = False
    batch_size: int = Field(default=2, ge=1, le=512)
    max_input_tokens: int = Field(default=8192, ge=8, le=32768)
    local_snapshot_path: str = Field(default="", min_length=0)
    network_fallback_allowed: bool = False

    @model_validator(mode="after")
    def _enforce_frozen_identity(self) -> EncoderConfig:
        if self.model_id != FROZEN_MODEL_ID:
            raise ValueError(
                f"model_id must be {FROZEN_MODEL_ID!r}, "
                f"got {self.model_id!r}"
            )
        if self.model_revision != FROZEN_MODEL_REVISION:
            raise ValueError(
                "model_revision must be the frozen BGE-M3 revision"
            )
        if self.dense_dimension != FROZEN_DENSE_DIMENSION:
            raise ValueError(
                f"dense_dimension must be {FROZEN_DENSE_DIMENSION}"
            )
        if self.dense_normalized is not True:
            raise ValueError("dense_normalized must be True")
        if self.network_fallback_allowed is not False:
            raise ValueError(
                "network_fallback_allowed must be False; the encoder "
                "must run offline against a local snapshot"
            )
        return self

    def config_hash(self) -> str:
        payload = json.dumps(
            {
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "dense_dimension": self.dense_dimension,
                "dense_normalized": self.dense_normalized,
                "device": self.device,
                "use_fp16": self.use_fp16,
                "batch_size": self.batch_size,
                "max_input_tokens": self.max_input_tokens,
                "local_snapshot_path": self.local_snapshot_path,
                "network_fallback_allowed": self.network_fallback_allowed,
                "embedder_version": EMBEDDER_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class DenseEmbedderProtocol(Protocol):
    """Protocol implemented by every Metadata-RAG dense encoder.

    The protocol exists so that the pipeline can be unit-tested with a
    deterministic fake encoder without loading BGE-M3. The concrete
    encoder and the fake encoder satisfy the same contract.
    """

    @property
    def config(self) -> EncoderConfig: ...

    def encode(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]: ...


def _validate_dense_vector(
    vector: tuple[float, ...], *, dense_dimension: int
) -> None:
    if len(vector) != dense_dimension:
        raise MetadataRAGEmbedderError(
            f"dense vector length {len(vector)} does not match "
            f"configured dense_dimension {dense_dimension}"
        )
    for value in vector:
        if not math.isfinite(value):
            raise MetadataRAGEmbedderError(
                "dense vector contains a non-finite value"
            )
    norm = math.sqrt(sum(v * v for v in vector))
    if abs(norm - 1.0) > _NORMALIZATION_TOLERANCE:
        raise MetadataRAGEmbedderError(
            f"dense vector is not L2-normalized (norm={norm})"
        )


def _load_bgem3_class() -> Any:
    """Dynamically import FlagEmbedding and return BGEM3FlagModel.

    The module is typed as ``Any`` because FlagEmbedding ships without
    type stubs. ``importlib`` is used so that the import is deferred
    until the first encode call.
    """
    module: Any = importlib.import_module("FlagEmbedding")
    return module.BGEM3FlagModel


class BgeM3DenseEncoder:
    """Concrete frozen BGE-M3 dense encoder.

    The model is loaded lazily on first call to ``encode``. The
    constructor is cheap and safe to call from tests, validators, and
    pipeline setup routines that do not perform actual embedding.
    """

    def __init__(self, config: EncoderConfig) -> None:
        if not isinstance(config, EncoderConfig):
            raise TypeError("config must be an EncoderConfig instance")
        self._config = config
        self._model: Any | None = None

    @property
    def config(self) -> EncoderConfig:
        return self._config

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        cfg = self._config
        if not cfg.local_snapshot_path:
            raise MetadataRAGEmbedderError(
                "local_snapshot_path must be set before encoding; the "
                "encoder runs offline against a local snapshot and "
                "never falls back to the network"
            )
        bgem3_cls = _load_bgem3_class()
        model = bgem3_cls(
            cfg.model_id,
            revision=cfg.model_revision,
            use_fp16=cfg.use_fp16,
            device=cfg.device,
            local_files_only=True,
        )
        self._model = model
        return model

    def encode(
        self, texts: Sequence[str]
    ) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        cfg = self._config
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise MetadataRAGEmbedderError(
                    f"texts[{index}] must be a non-empty string"
                )
        model = self._ensure_loaded()

        # BGEM3FlagModel.encode returns a dict with a "dense_vecs"
        # entry whose rows are numpy float32 arrays.
        out = model.encode(
            list(texts),
            batch_size=cfg.batch_size,
            max_length=cfg.max_input_tokens,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        rows: list[tuple[float, ...]] = []
        for row in dense:
            vector = tuple(float(x) for x in row)
            _validate_dense_vector(
                vector, dense_dimension=cfg.dense_dimension
            )
            rows.append(vector)
        if len(rows) != len(texts):
            raise MetadataRAGEmbedderError(
                "encoder returned a different number of rows than inputs"
            )
        return tuple(rows)


# --- Embedding records ------------------------------------------------------


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class MetadataRAGBaseModel(BaseModel):
    """Strict, frozen base model for Metadata-RAG embedding records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )


class MetadataRAGDocumentEmbedding(MetadataRAGBaseModel):
    """Dense embedding of one canonical document's text.

    The record links the embedding back to the frozen metadata
    record and to the canonical document. It carries the encoder
    config hash so that records cannot be silently mixed across
    encoder identities.
    """

    record_kind: Literal["document_embedding"] = "document_embedding"
    embedding_version: str = EMBEDDER_VERSION

    metadata_record_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    document_order: int = Field(ge=0)

    source_text_sha256: str = Field(min_length=1)

    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    embedding_config_hash: str = Field(min_length=1)

    dense_vector: tuple[float, ...]
    dense_dimension: int = Field(ge=1)
    dense_normalized: bool

    @model_validator(mode="after")
    def _enforce_shape(self) -> MetadataRAGDocumentEmbedding:
        if self.dense_dimension != FROZEN_DENSE_DIMENSION:
            raise ValueError(
                f"dense_dimension must be {FROZEN_DENSE_DIMENSION}"
            )
        if self.dense_normalized is not True:
            raise ValueError("dense_normalized must be True")
        if self.model_id != FROZEN_MODEL_ID:
            raise ValueError(f"model_id must be {FROZEN_MODEL_ID!r}")
        if self.model_revision != FROZEN_MODEL_REVISION:
            raise ValueError(
                "model_revision must be the frozen BGE-M3 revision"
            )
        if len(self.dense_vector) != self.dense_dimension:
            raise ValueError(
                "dense_vector length does not match dense_dimension"
            )
        return self


class MetadataRAGQueryEmbedding(MetadataRAGBaseModel):
    """Dense embedding of one query text.

    The record links the embedding back to the canonical query_id.
    It carries only the query_id and the query text SHA-256; it does
    not carry the query text itself, so a persisted embedding file
    cannot be mined for ground truth.
    """

    record_kind: Literal["query_embedding"] = "query_embedding"
    embedding_version: str = EMBEDDER_VERSION

    dataset_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)

    query_text_sha256: str = Field(min_length=1)

    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    embedding_config_hash: str = Field(min_length=1)

    dense_vector: tuple[float, ...]
    dense_dimension: int = Field(ge=1)
    dense_normalized: bool

    @model_validator(mode="after")
    def _enforce_shape(self) -> MetadataRAGQueryEmbedding:
        if self.dense_dimension != FROZEN_DENSE_DIMENSION:
            raise ValueError(
                f"dense_dimension must be {FROZEN_DENSE_DIMENSION}"
            )
        if self.dense_normalized is not True:
            raise ValueError("dense_normalized must be True")
        if self.model_id != FROZEN_MODEL_ID:
            raise ValueError(f"model_id must be {FROZEN_MODEL_ID!r}")
        if self.model_revision != FROZEN_MODEL_REVISION:
            raise ValueError(
                "model_revision must be the frozen BGE-M3 revision"
            )
        if len(self.dense_vector) != self.dense_dimension:
            raise ValueError(
                "dense_vector length does not match dense_dimension"
            )
        return self


def build_document_embedding(
    *,
    config: EncoderConfig,
    dataset_id: str,
    doc_id: str,
    document_order: int,
    metadata_record_id: str,
    source_text: str,
    dense_vector: tuple[float, ...],
) -> MetadataRAGDocumentEmbedding:
    """Build one document embedding record and validate it."""
    if not source_text.strip():
        raise MetadataRAGEmbedderError("source_text cannot be empty")
    _validate_dense_vector(
        dense_vector, dense_dimension=config.dense_dimension
    )
    return MetadataRAGDocumentEmbedding(
        metadata_record_id=metadata_record_id,
        dataset_id=dataset_id,
        doc_id=doc_id,
        document_order=document_order,
        source_text_sha256=_sha256_text(source_text),
        model_id=config.model_id,
        model_revision=config.model_revision,
        embedding_config_hash=config.config_hash(),
        dense_vector=dense_vector,
        dense_dimension=config.dense_dimension,
        dense_normalized=config.dense_normalized,
    )


def build_query_embedding(
    *,
    config: EncoderConfig,
    dataset_id: str,
    query_id: str,
    query_text: str,
    dense_vector: tuple[float, ...],
) -> MetadataRAGQueryEmbedding:
    """Build one query embedding record and validate it."""
    if not query_text.strip():
        raise MetadataRAGEmbedderError("query_text cannot be empty")
    _validate_dense_vector(
        dense_vector, dense_dimension=config.dense_dimension
    )
    return MetadataRAGQueryEmbedding(
        dataset_id=dataset_id,
        query_id=query_id,
        query_text_sha256=_sha256_text(query_text),
        model_id=config.model_id,
        model_revision=config.model_revision,
        embedding_config_hash=config.config_hash(),
        dense_vector=dense_vector,
        dense_dimension=config.dense_dimension,
        dense_normalized=config.dense_normalized,
    )
