"""Bounded orchestration helpers for the LongRAG synthetic acceptance run."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from pydantic import Field

from sah_rag.artifacts import ArtifactBaseModel
from baselines.longrag import (
    LongRAGSourceSegment,
    LongUnit,
)
from baselines.longrag_embeddings import (
    LongUnitEmbeddingRecord,
)
from sah_rag.canonical.models import (
    CanonicalDocument,
    CanonicalExample,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _whitespace_token_count(text: str) -> int:
    return len(text.split())


class ProjectedQuery(ArtifactBaseModel):
    """Retrieval-time query projection.

    Deliberately excludes doc_ids, relevant_ids, answers, and any
    content-group metadata so that the retriever cannot see ground truth.
    """

    query_id: str
    query_text: str


class LongRAGEvaluationCase(ArtifactBaseModel):
    """One evaluation case for the controlled synthetic acceptance run."""

    query_id: str
    ranked_doc_ids: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]
    metadata: dict[str, object] = Field(default_factory=dict)


def build_source_segments(
    documents: Sequence[CanonicalDocument],
    *,
    dataset_id: str,
) -> tuple[LongRAGSourceSegment, ...]:
    """Convert canonical documents into deterministic LongRAG source segments.

    Each document becomes exactly one segment with a deterministic
    parent_id (doc_id + ':p0') and chunk_id (doc_id + ':c0').
    Token count is the deterministic whitespace-token count.
    """
    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be empty.")

    segments: list[LongRAGSourceSegment] = []
    for order, doc in enumerate(documents):
        text = doc.text_normalized
        token_count = _whitespace_token_count(text)
        if token_count == 0:
            raise ValueError(f"Document {doc.doc_id} is empty.")
        segments.append(
            LongRAGSourceSegment(
                dataset_id=dataset_id,
                doc_id=doc.doc_id,
                parent_id=f"{doc.doc_id}:p0",
                chunk_id=f"{doc.doc_id}:c0",
                document_order=order,
                parent_order=0,
                chunk_order=0,
                text=text,
                token_count=token_count,
                metadata={
                    "source": "canonical_document",
                    "source_content_group_id": str(
                        doc.metadata.get("source_content_group_id", "")
                    ),
                },
            )
        )
    return tuple(segments)


def project_queries(
    examples: Sequence[CanonicalExample],
) -> tuple[ProjectedQuery, ...]:
    """Extract only query_id and question text for retrieval."""
    projected: list[ProjectedQuery] = []
    for example in examples:
        projected.append(
            ProjectedQuery(
                query_id=example.query.query_id,
                query_text=example.query.question,
            )
        )
    return tuple(projected)


def extract_relevant_ids(
    examples: Sequence[CanonicalExample],
) -> dict[str, tuple[str, ...]]:
    """Load ground-truth relevant ids AFTER retrieval has completed."""
    mapping: dict[str, tuple[str, ...]] = {}
    for example in examples:
        mapping[example.query.query_id] = tuple(
            example.ground_truth.relevant_ids
        )
    return mapping


def build_embedding_record(
    *,
    long_unit: LongUnit,
    condition_id: str,
    target_token_size: int,
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    embedding_config_hash: str,
    dense_vector: tuple[float, ...],
    dense_dimension: int,
    special_token_count: int,
    maximum_model_input_tokens: int,
    source_units_file_sha256: str,
    adapter_id: str,
    adapter_version: str,
    source_architecture_commit: str,
) -> LongUnitEmbeddingRecord:
    """Build one dense-only LongUnitEmbeddingRecord from a LongUnit."""
    rendered = long_unit.token_count
    actual = rendered + special_token_count
    if actual > maximum_model_input_tokens:
        raise ValueError(
            f"Long unit {long_unit.long_unit_id} exceeds the model input "
            f"budget ({actual} > {maximum_model_input_tokens})."
        )
    if len(dense_vector) != dense_dimension:
        raise ValueError(
            "dense_vector length does not match dense_dimension."
        )
    return LongUnitEmbeddingRecord(
        long_unit_id=long_unit.long_unit_id,
        condition_id=condition_id,
        dataset_id=long_unit.dataset_id,
        construction_policy=long_unit.construction_policy,
        target_token_size=target_token_size,
        maximum_model_input_tokens=maximum_model_input_tokens,
        unit_order=long_unit.unit_order,
        rendered_token_count=rendered,
        special_token_count=special_token_count,
        actual_model_token_count=actual,
        was_truncated=False,
        member_doc_ids=long_unit.member_doc_ids,
        member_parent_ids=long_unit.member_parent_ids,
        member_chunk_ids=long_unit.member_chunk_ids,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        embedding_config_hash=embedding_config_hash,
        dense_vector=dense_vector,
        dense_dimension=dense_dimension,
        dense_precision="float32",
        dense_normalized=True,
        source_unit_sha256=_sha256_text(long_unit.text),
        source_units_file_sha256=source_units_file_sha256,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        source_architecture="LongRAG",
        source_architecture_commit=source_architecture_commit,
        interface_validation=True,
        paper_faithful_result=False,
    )


def build_evaluation_case(
    *,
    query_id: str,
    ranked_doc_ids: tuple[str, ...],
    relevant_doc_ids: tuple[str, ...],
) -> LongRAGEvaluationCase:
    """Build one evaluation case. Never called during retrieval."""
    if not query_id.strip():
        raise ValueError("query_id cannot be empty.")
    return LongRAGEvaluationCase(
        query_id=query_id,
        ranked_doc_ids=ranked_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        metadata={},
    )