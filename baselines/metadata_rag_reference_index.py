"""Exact in-memory dense reference index for the Metadata-RAG comparator.

This module IS the production retrieval substrate for Metadata-RAG.
Per baseline_audit/metadata_rag_clean_room_spec_amendment_02.yaml,
storage is IN_MEMORY_EXACT_DENSE with equivalence_test: NOT_REQUIRED,
because the paper does not pin a storage backend and the project
intentionally avoids approximate search.

Windowing
---------
Per amendment 04, one canonical document may be split into one or
more windows. Each window has its own dense vector, its own metadata
record, and its own window ``doc_id``. Every window carries the
canonical document's identity as ``lineage_doc_id``, which is the
deduplication key used by the document expansion stage.

The index therefore stores, per entry:

  * ``doc_id``        -- window identifier
  * ``lineage_doc_id`` -- canonical document identifier
  * ``window_index``  -- 0-based position within the parent document
  * ``chunk_id``      -- deterministic chunk identifier of the window

The index performs exact, exhaustive, unpruned cosine similarity
search over L2-normalized vectors. It uses no approximate neighbor
search, no quantization, no HNSW, and no candidate pruning. Ties are
broken deterministically by ascending window ``doc_id``, which
naturally orders windows of one document in window-index order.

Ground truth is never read by this module.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from baselines.metadata_rag_adapter import build_chunk_id
from baselines.metadata_rag_embedder import (
    FROZEN_DENSE_DIMENSION,
    MetadataRAGDocumentEmbedding,
)
from baselines.metadata_rag_metadata import MetadataRecord
from baselines.metadata_rag_state import MetadataRAGState

REFERENCE_INDEX_VERSION: Final[str] = "0.2.0"

_NORMALIZATION_TOLERANCE: Final[float] = 1e-3

_MatrixF32 = npt.NDArray[np.float32]


class MetadataRAGReferenceIndexError(ValueError):
    """Raised when the index violates its frozen contract."""


class MetadataRAGBaseModel(BaseModel):
    """Strict, frozen base model for reference-index artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )


class DenseIndexEntry(MetadataRAGBaseModel):
    """One entry in the frozen exact dense index.

    Fields:
        dataset_id: canonical dataset identifier.
        lineage_doc_id: canonical document identifier.
        doc_id: window identifier. Equals ``lineage_doc_id`` for a
            single-window document; ``<lineage_doc_id>:w<i>`` otherwise.
        window_index: zero-based window index within the parent
            document.
        metadata_record_id: content-addressable metadata record ID.
        document_order: zero-based position in the flattened window
            stream.
        dense_vector: L2-normalized dense vector of the window text.
        metadata: the frozen MetadataRecord derived from the window
            text. Its ``doc_id`` field refers to the canonical document.
    """

    dataset_id: str = Field(min_length=1)
    lineage_doc_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    window_index: int = Field(default=0, ge=0)
    metadata_record_id: str = Field(min_length=1)
    document_order: int = Field(ge=0)
    dense_vector: tuple[float, ...]
    metadata: MetadataRecord

    @model_validator(mode="after")
    def _enforce_shape(self) -> DenseIndexEntry:
        if len(self.dense_vector) != FROZEN_DENSE_DIMENSION:
            raise ValueError(
                "dense_vector length must be "
                f"{FROZEN_DENSE_DIMENSION}"
            )
        for value in self.dense_vector:
            if not math.isfinite(value):
                raise ValueError(
                    "dense_vector contains a non-finite value"
                )
        norm = math.sqrt(sum(v * v for v in self.dense_vector))
        if abs(norm - 1.0) > _NORMALIZATION_TOLERANCE:
            raise ValueError(
                f"dense_vector is not L2-normalized (norm={norm})"
            )
        if self.metadata.source_kind != "document":
            raise ValueError(
                "DenseIndexEntry.metadata.source_kind must be 'document'"
            )
        if self.metadata.doc_id != self.lineage_doc_id:
            raise ValueError(
                "DenseIndexEntry.metadata.doc_id must equal "
                "lineage_doc_id"
            )
        return self


class DenseSearchHit(MetadataRAGBaseModel):
    """One hit returned by the exact dense reference index.

    The hit carries the window identity, the lineage (canonical)
    document identity, the deterministic chunk identifier, the
    metadata record identity, the 1-based rank, and the raw cosine
    score (HIGHER_IS_BETTER). The retrieval pipeline is responsible
    for any downstream fusion; the hit never carries fused scores.
    """

    doc_id: str = Field(min_length=1)
    lineage_doc_id: str = Field(min_length=1)
    window_index: int = Field(default=0, ge=0)
    chunk_id: str = Field(min_length=1)
    metadata_record_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    score: float
    score_semantics: Literal["HIGHER_IS_BETTER"] = "HIGHER_IS_BETTER"

    @model_validator(mode="after")
    def _enforce_shape(self) -> DenseSearchHit:
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        return self


def _stack_matrix(
    entries: Sequence[DenseIndexEntry],
) -> _MatrixF32:
    rows = [list(e.dense_vector) for e in entries]
    arr = np.asarray(rows, dtype=np.float32)
    return arr


class ReferenceDenseIndex:
    """Exact in-memory dense reference index.

    Construction is O(N * D) and search is O(|candidates| * D). The
    index is immutable after construction.
    """

    def __init__(self, entries: Sequence[DenseIndexEntry]) -> None:
        entries_tuple = tuple(entries)
        if not entries_tuple:
            raise MetadataRAGReferenceIndexError(
                "ReferenceDenseIndex requires at least one entry"
            )
        dataset_ids = {e.dataset_id for e in entries_tuple}
        if len(dataset_ids) != 1:
            raise MetadataRAGReferenceIndexError(
                "all entries must share the same dataset_id; "
                f"observed {sorted(dataset_ids)}"
            )
        doc_ids = [e.doc_id for e in entries_tuple]
        if len(set(doc_ids)) != len(doc_ids):
            raise MetadataRAGReferenceIndexError(
                "duplicate doc_id in entries"
            )
        expected_order = list(range(len(entries_tuple)))
        observed_order = [e.document_order for e in entries_tuple]
        if observed_order != expected_order:
            raise MetadataRAGReferenceIndexError(
                "entries must be ordered contiguously by document_order"
            )

        self._dataset_id: str = next(iter(dataset_ids))
        self._entries: tuple[DenseIndexEntry, ...] = entries_tuple
        self._matrix: _MatrixF32 = _stack_matrix(entries_tuple)
        self._doc_ids: tuple[str, ...] = tuple(doc_ids)
        self._pos_by_doc_id: dict[str, int] = {
            doc_id: pos for pos, doc_id in enumerate(doc_ids)
        }

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self._doc_ids

    @property
    def dense_dimension(self) -> int:
        return int(self._matrix.shape[1])

    def entries(self) -> tuple[DenseIndexEntry, ...]:
        return self._entries

    def metadata_for(self, doc_id: str) -> MetadataRecord:
        pos = self._pos_by_doc_id.get(doc_id)
        if pos is None:
            raise MetadataRAGReferenceIndexError(
                f"unknown doc_id: {doc_id!r}"
            )
        return self._entries[pos].metadata

    def lineage_for(self, doc_id: str) -> str:
        pos = self._pos_by_doc_id.get(doc_id)
        if pos is None:
            raise MetadataRAGReferenceIndexError(
                f"unknown doc_id: {doc_id!r}"
            )
        return self._entries[pos].lineage_doc_id

    # -- internal helpers ---------------------------------------------------

    def _validate_query(
        self, query_vector: Sequence[float]
    ) -> npt.NDArray[np.float32]:
        query = np.asarray(list(query_vector), dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.dense_dimension:
            raise MetadataRAGReferenceIndexError(
                f"query_vector must have shape "
                f"({self.dense_dimension},); "
                f"observed {tuple(query.shape)}"
            )
        for value in query.tolist():
            if not math.isfinite(value):
                raise MetadataRAGReferenceIndexError(
                    "query_vector contains a non-finite value"
                )
        norm = float(np.linalg.norm(query))
        if abs(norm - 1.0) > _NORMALIZATION_TOLERANCE:
            raise MetadataRAGReferenceIndexError(
                f"query_vector is not L2-normalized (norm={norm})"
            )
        return query

    def _candidate_positions(
        self, candidate_doc_ids: Iterable[str]
    ) -> npt.NDArray[np.int64]:
        positions: list[int] = []
        seen: set[str] = set()
        for doc_id in candidate_doc_ids:
            if doc_id in seen:
                continue
            pos = self._pos_by_doc_id.get(doc_id)
            if pos is None:
                raise MetadataRAGReferenceIndexError(
                    f"candidate doc_id not present in index: {doc_id!r}"
                )
            positions.append(pos)
            seen.add(doc_id)
        positions.sort()
        return np.asarray(positions, dtype=np.int64)

    # -- search -------------------------------------------------------------

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        candidate_doc_ids: frozenset[str] | None = None,
    ) -> tuple[DenseSearchHit, ...]:
        """Exact cosine top-k retrieval.

        Args:
            query_vector: L2-normalized dense vector of length 1024.
            top_k: requested number of hits (>= 1).
            candidate_doc_ids: optional restriction to a set of
                window doc_ids. When the restriction is empty,
                returns the empty tuple. When None, searches over the
                full corpus.

        Returns:
            A tuple of hits, ranks starting at 1, contiguous, sorted
            by (-score, window_doc_id) ascending.
        """
        if top_k < 1:
            raise MetadataRAGReferenceIndexError(
                "top_k must be >= 1"
            )
        query = self._validate_query(query_vector)

        if candidate_doc_ids is None:
            positions = np.arange(self.size, dtype=np.int64)
        else:
            positions = self._candidate_positions(candidate_doc_ids)
            if positions.size == 0:
                return ()

        subset = self._matrix[positions]
        scores = subset @ query

        order = sorted(
            range(len(positions)),
            key=lambda i: (
                -float(scores[i]),
                self._doc_ids[int(positions[i])],
            ),
        )
        limit = min(top_k, len(order))
        hits: list[DenseSearchHit] = []
        for rank, i in enumerate(order[:limit], start=1):
            pos = int(positions[i])
            entry = self._entries[pos]
            hits.append(
                DenseSearchHit(
                    doc_id=entry.doc_id,
                    lineage_doc_id=entry.lineage_doc_id,
                    window_index=entry.window_index,
                    chunk_id=build_chunk_id(
                        entry.lineage_doc_id, entry.window_index
                    ),
                    metadata_record_id=entry.metadata_record_id,
                    rank=rank,
                    score=float(scores[i]),
                )
            )
        return tuple(hits)


def build_reference_dense_index(
    *,
    state: MetadataRAGState,
    embeddings: Sequence[MetadataRAGDocumentEmbedding],
) -> ReferenceDenseIndex:
    """Join a Metadata-RAG state and its window embeddings.

    The state and the embeddings must form a strict bijection:

      * every state window has exactly one embedding,
      * every embedding refers to a state window,
      * the (dataset_id, doc_id, metadata_record_id, document_order)
        tuple matches on both sides.

    Any deviation raises MetadataRAGReferenceIndexError.
    """
    state_by_doc: dict[str, object] = {
        d.doc_id: d for d in state.documents
    }
    if len(state_by_doc) != len(state.documents):
        raise MetadataRAGReferenceIndexError(
            "state contains duplicate doc_id values"
        )

    embeddings_by_doc: dict[str, MetadataRAGDocumentEmbedding] = {}
    for candidate_embedding in embeddings:
        if candidate_embedding.doc_id in embeddings_by_doc:
            raise MetadataRAGReferenceIndexError(
                "duplicate embedding for doc_id: "
                f"{candidate_embedding.doc_id!r}"
            )
        embeddings_by_doc[candidate_embedding.doc_id] = (
            candidate_embedding
        )

    orphans_emb = set(embeddings_by_doc) - set(state_by_doc)
    if orphans_emb:
        raise MetadataRAGReferenceIndexError(
            f"embeddings without matching state window: "
            f"{sorted(orphans_emb)}"
        )

    entries: list[DenseIndexEntry] = []
    for doc in state.documents:
        maybe_emb = embeddings_by_doc.get(doc.doc_id)
        if maybe_emb is None:
            raise MetadataRAGReferenceIndexError(
                f"missing embedding for doc_id: {doc.doc_id!r}"
            )
        if maybe_emb.dataset_id != state.dataset_id:
            raise MetadataRAGReferenceIndexError(
                f"embedding {maybe_emb.doc_id!r} dataset_id "
                f"{maybe_emb.dataset_id!r} does not match state "
                f"dataset_id {state.dataset_id!r}"
            )
        if maybe_emb.metadata_record_id != doc.metadata_record_id:
            raise MetadataRAGReferenceIndexError(
                f"embedding {maybe_emb.doc_id!r} metadata_record_id "
                f"mismatch: {maybe_emb.metadata_record_id!r} != "
                f"{doc.metadata_record_id!r}"
            )
        if maybe_emb.document_order != doc.document_order:
            raise MetadataRAGReferenceIndexError(
                f"embedding {maybe_emb.doc_id!r} document_order "
                f"mismatch: {maybe_emb.document_order} != "
                f"{doc.document_order}"
            )
        entries.append(
            DenseIndexEntry(
                dataset_id=state.dataset_id,
                lineage_doc_id=doc.lineage_doc_id,
                doc_id=doc.doc_id,
                window_index=doc.window_index,
                metadata_record_id=doc.metadata_record_id,
                document_order=doc.document_order,
                dense_vector=maybe_emb.dense_vector,
                metadata=doc.metadata,
            )
        )

    return ReferenceDenseIndex(tuple(entries))
