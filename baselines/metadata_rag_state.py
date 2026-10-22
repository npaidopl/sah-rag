"""In-memory indexed state for the Metadata-RAG comparator.

The state is a deterministic, ordered collection of indexed documents.
It carries no embeddings, no storage backend, and no query-time
structure. It is the input to the embedding stage and to any
persistence backend, and it can be validated and reopened
deterministically.

Frozen specification:
    baseline_audit/metadata_rag_clean_room_spec.yaml
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field, model_validator

from baselines.metadata_rag_adapter import (
    INDEXED_DOCUMENT_VERSION,
    MetadataRAGBaseModel,
    MetadataRAGIndexedDocument,
)
from baselines.metadata_rag_metadata import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
)

STATE_VERSION = "0.1.0"


class MetadataRAGState(MetadataRAGBaseModel):
    """Deterministic in-memory state for one dataset.

    Invariants:
        - ``documents`` is non-empty.
        - ``documents`` is ordered by ``document_order`` ascending.
        - ``document_order`` values are contiguous starting at 0.
        - No duplicate ``doc_id`` values within the state.
        - All documents share the same dataset, schema version, and
          extractor version.
    """

    dataset_id: str = Field(min_length=1)
    documents: tuple[MetadataRAGIndexedDocument, ...]
    schema_version: str = SCHEMA_VERSION
    extractor_version: str = EXTRACTOR_VERSION
    state_version: str = STATE_VERSION

    @model_validator(mode="after")
    def _check_state_invariants(self) -> MetadataRAGState:
        docs = self.documents
        if not docs:
            raise ValueError("state.documents cannot be empty")

        for index, doc in enumerate(docs):
            if doc.document_order != index:
                raise ValueError(
                    "state.documents must be ordered contiguously by "
                    f"document_order; index {index} has "
                    f"document_order {doc.document_order}"
                )

        seen: set[str] = set()
        for doc in docs:
            if doc.doc_id in seen:
                raise ValueError(
                    f"state.documents contains duplicate doc_id: "
                    f"{doc.doc_id!r}"
                )
            seen.add(doc.doc_id)

        for doc in docs:
            if doc.dataset_id != self.dataset_id:
                raise ValueError(
                    f"document {doc.doc_id!r} dataset_id "
                    f"{doc.dataset_id!r} does not match state "
                    f"dataset_id {self.dataset_id!r}"
                )
            if doc.schema_version != self.schema_version:
                raise ValueError(
                    f"document {doc.doc_id!r} schema_version mismatch"
                )
            if doc.extractor_version != self.extractor_version:
                raise ValueError(
                    f"document {doc.doc_id!r} extractor_version mismatch"
                )
            if doc.record_version != INDEXED_DOCUMENT_VERSION:
                raise ValueError(
                    f"document {doc.doc_id!r} record_version mismatch"
                )

        return self


def build_state(
    documents: Iterable[MetadataRAGIndexedDocument],
    *,
    dataset_id: str,
) -> MetadataRAGState:
    """Build a validated state from an ordered document sequence."""
    doc_tuple = tuple(documents)
    if not doc_tuple:
        raise ValueError("build_state requires at least one document")
    return MetadataRAGState(
        dataset_id=dataset_id,
        documents=doc_tuple,
    )
