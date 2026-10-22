"""Canonical-to-Metadata-RAG adapter.

Converts canonical documents and examples into the deterministic,
frozen intermediate representation used by the Metadata-RAG comparator.

Frozen specification:
    baseline_audit/metadata_rag_clean_room_spec.yaml
Frozen amendments:
    baseline_audit/metadata_rag_clean_room_spec_amendment_01.yaml
    baseline_audit/metadata_rag_clean_room_spec_amendment_02.yaml
    baseline_audit/metadata_rag_clean_room_spec_amendment_03.yaml
    baseline_audit/metadata_rag_clean_room_spec_amendment_04.yaml

Windowing
---------
Amendment 04 fixes a deterministic oversize-document windowing policy.
A canonical document whose token count is at or below the frozen
window budget (8190 content tokens) produces exactly one window with
``window_index = 0`` and ``doc_id = lineage_doc_id = canonical doc_id``.
A canonical document above the budget is split into N contiguous,
exhaustive windows. Each window carries ``lineage_doc_id`` equal to the
canonical doc_id and ``doc_id`` equal to ``<lineage_doc_id>:w<i>``.

The metadata record is always derived from the text of the window
itself, so its ``doc_id`` field refers to the canonical document while
its content reflects the specific window's slice.

This module never reads query.doc_ids, query.metadata,
ground_truth.relevant_ids, ground_truth.answers, or any other
ground-truth surface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from baselines.metadata_rag_metadata import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    MetadataRecord,
    build_document_metadata,
)
from baselines.metadata_rag_windowing import (
    SingleWindowSplitter,
    WindowSplitterProtocol,
)
from sah_rag.canonical.models import (
    CanonicalDocument,
    CanonicalExample,
)

ADAPTER_VERSION = "0.2.0"
INDEXED_DOCUMENT_VERSION = "0.2.0"


class MetadataRAGBaseModel(BaseModel):
    """Strict, frozen base model for Metadata-RAG artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )


class MetadataRAGProjectedQuery(MetadataRAGBaseModel):
    """Retrieval-time query projection.

    Contains exactly ``query_id`` and ``query_text``. Any other field
    would leak ground truth into retrieval and is prohibited by the
    frozen clean-room spec.
    """

    query_id: str
    query_text: str


class MetadataRAGIndexedDocument(MetadataRAGBaseModel):
    """One canonical document window, as indexed by Metadata-RAG.

    Fields:
        metadata_record_id: content-addressable identifier of the
            metadata record for this window.
        dataset_id: canonical dataset identifier.
        lineage_doc_id: canonical document identifier. This is the
            deduplication key used by the document expansion stage.
        doc_id: window identifier. Equal to ``lineage_doc_id`` for a
            single-window document; ``<lineage_doc_id>:w<i>`` otherwise.
        window_index: zero-based window index within the parent
            document.
        parent_id: deterministic parent identifier. Same for every
            window of one ``lineage_doc_id``.
        chunk_id: deterministic chunk identifier for this window.
        document_order: zero-based position in the flattened stream
            of windows.
        text_normalized: the exact slice of the parent document's
            normalized text assigned to this window.
        token_count: whitespace token count of ``text_normalized``.
        metadata: the frozen MetadataRecord derived from this window's
            text. Its ``doc_id`` field refers to the canonical
            document (the provenance tier).
    """

    metadata_record_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    lineage_doc_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    window_index: int = Field(default=0, ge=0)
    parent_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document_order: int = Field(ge=0)
    text_normalized: str = Field(min_length=1)
    token_count: int = Field(ge=0)
    metadata: MetadataRecord
    schema_version: str = SCHEMA_VERSION
    extractor_version: str = EXTRACTOR_VERSION
    adapter_version: str = ADAPTER_VERSION

    record_type: Literal["metadata_rag_indexed_document"] = (
        "metadata_rag_indexed_document"
    )
    record_version: str = INDEXED_DOCUMENT_VERSION


def build_parent_id(lineage_doc_id: str) -> str:
    """Deterministic parent identifier for one canonical document.

    The parent is the canonical document itself, not the window. Every
    window of one document shares the same parent_id.
    """
    if not lineage_doc_id.strip():
        raise ValueError("lineage_doc_id cannot be empty")
    return f"{lineage_doc_id}:p0"


def build_chunk_id(
    lineage_doc_id: str, window_index: int = 0
) -> str:
    """Deterministic chunk identifier for one window."""
    if not lineage_doc_id.strip():
        raise ValueError("lineage_doc_id cannot be empty")
    if window_index < 0:
        raise ValueError("window_index cannot be negative")
    return f"{lineage_doc_id}:c{window_index}"


def build_window_id(lineage_doc_id: str, window_index: int) -> str:
    """Deterministic window identifier for a non-first window.

    This is only used when the parent document is split into more than
    one window. Single-window documents do not use a window_id; their
    doc_id equals the lineage_doc_id.
    """
    if not lineage_doc_id.strip():
        raise ValueError("lineage_doc_id cannot be empty")
    if window_index < 0:
        raise ValueError("window_index cannot be negative")
    return f"{lineage_doc_id}:w{window_index}"


def _derive_doc_id(
    lineage_doc_id: str, window_index: int, n_windows: int
) -> str:
    """Apply the frozen single-window shortcut.

    For a parent document that fits within the window budget, the
    window's doc_id is exactly the lineage_doc_id, so the frozen
    un-windowed behaviour is preserved byte for byte. For multi-window
    parents, the window's doc_id follows the ``:wN`` pattern fixed by
    amendment 04.
    """
    if n_windows == 1 and window_index == 0:
        return lineage_doc_id
    return build_window_id(lineage_doc_id, window_index)


def build_metadata_record_id(
    *,
    dataset_id: str,
    doc_id: str,
    metadata: MetadataRecord,
) -> str:
    """Content-addressable, deterministic metadata record identifier.

    The ID depends only on the frozen inputs that define the record.
    Any change to the metadata content or to the frozen schema /
    extractor versions produces a different ID.
    """
    payload = json.dumps(
        {
            "dataset_id": dataset_id,
            "doc_id": doc_id,
            "schema_version": metadata.schema_version,
            "extractor_version": metadata.extractor_version,
            "capitalized_phrases": list(metadata.capitalized_phrases),
            "year_mentions": list(metadata.year_mentions),
            "numeric_tokens": list(metadata.numeric_tokens),
            "char_length": metadata.char_length,
            "word_count": metadata.word_count,
            "length_bucket": metadata.length_bucket,
            "source_content_group_id": metadata.source_content_group_id,
            "title": metadata.title,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"mdrec-{digest}"


def _whitespace_token_count(text: str) -> int:
    return len(text.split())


def build_indexed_document(
    *,
    document: CanonicalDocument,
    document_order: int,
    dataset_id: str,
    window_index: int = 0,
    n_windows: int = 1,
    window_text: str | None = None,
) -> MetadataRAGIndexedDocument:
    """Convert one canonical document (or one window of it) into an
    indexed document.

    Default arguments model the single-window case, in which the
    frozen windowing amendment is a no-op and the produced record is
    identical to the pre-amendment behaviour.
    """
    if document_order < 0:
        raise ValueError("document_order cannot be negative")
    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be empty")
    if window_index < 0:
        raise ValueError("window_index cannot be negative")
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    if window_index >= n_windows:
        raise ValueError("window_index must be < n_windows")

    lineage_doc_id = document.doc_id
    effective_text = (
        window_text if window_text is not None else document.text_normalized
    )
    if not effective_text.strip():
        raise ValueError("window_text cannot be empty")

    source_dataset_id_raw = document.metadata.get("source_dataset_id")
    source_dataset_id = (
        str(source_dataset_id_raw)
        if source_dataset_id_raw is not None
        else document.dataset
    )

    source_content_group_id_raw = document.metadata.get(
        "source_content_group_id"
    )
    source_content_group_id = (
        str(source_content_group_id_raw)
        if source_content_group_id_raw is not None
        else None
    )

    metadata = build_document_metadata(
        dataset=document.dataset,
        doc_id=lineage_doc_id,
        source_dataset_id=source_dataset_id,
        text_normalized=effective_text,
        source_content_group_id=source_content_group_id,
        title=document.title,
    )

    doc_id = _derive_doc_id(lineage_doc_id, window_index, n_windows)

    record_id = build_metadata_record_id(
        dataset_id=dataset_id,
        doc_id=doc_id,
        metadata=metadata,
    )

    return MetadataRAGIndexedDocument(
        metadata_record_id=record_id,
        dataset_id=dataset_id,
        lineage_doc_id=lineage_doc_id,
        doc_id=doc_id,
        window_index=window_index,
        parent_id=build_parent_id(lineage_doc_id),
        chunk_id=build_chunk_id(lineage_doc_id, window_index),
        document_order=document_order,
        text_normalized=effective_text,
        token_count=_whitespace_token_count(effective_text),
        metadata=metadata,
    )


def build_indexed_documents(
    documents: Iterable[CanonicalDocument],
    *,
    dataset_id: str,
    splitter: WindowSplitterProtocol | None = None,
) -> tuple[MetadataRAGIndexedDocument, ...]:
    """Convert a canonical document stream into indexed windows.

    Windows are emitted in source order: for each canonical document,
    its windows appear in ascending ``window_index`` order, and the
    flattened ``document_order`` increases by 1 per window.

    The default splitter is the deterministic single-window splitter,
    under which this function is byte-for-byte equivalent to the
    pre-amendment behaviour. Real-dataset runs inject the pinned BGE-M3
    splitter via ``metadata_rag_windowing.build_bge_m3_window_splitter``.
    """
    effective_splitter: WindowSplitterProtocol = (
        splitter if splitter is not None else SingleWindowSplitter()
    )

    indexed: list[MetadataRAGIndexedDocument] = []
    flattened_order = 0
    for doc in documents:
        spans = effective_splitter.split(doc.text_normalized)
        n_windows = len(spans)
        for span in spans:
            window_text = doc.text_normalized[
                span.char_start : span.char_end
            ]
            indexed.append(
                build_indexed_document(
                    document=doc,
                    document_order=flattened_order,
                    dataset_id=dataset_id,
                    window_index=span.window_index,
                    n_windows=n_windows,
                    window_text=window_text,
                )
            )
            flattened_order += 1
    return tuple(indexed)


def project_queries(
    examples: Sequence[CanonicalExample],
) -> tuple[MetadataRAGProjectedQuery, ...]:
    """Project canonical examples to the retrieval-only query surface.

    Only ``query.query_id`` and ``query.question`` are read. Any
    other field on ``query`` or ``ground_truth`` is never touched.
    """
    return tuple(
        MetadataRAGProjectedQuery(
            query_id=example.query.query_id,
            query_text=example.query.question,
        )
        for example in examples
    )
