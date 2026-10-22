"""Canonical envelope adapter: pilot envelope -> sah_rag canonical.

The adapter reads the immutable Step 78 pilot envelope and emits a
derived canonical envelope in the shape that ``sah_rag`` expects.

Nothing is executed. No parser, chunker, embedding, retrieval,
evaluation, or generation runs. Source artifacts are never modified.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sah_rag.canonical.models import (
    CanonicalDocument,
    CanonicalExample,
    CanonicalGroundTruth,
    CanonicalQuery,
    GroundTruthType,
)
from datasets.pilot_synthetic import (
    PilotCanonicalChunk,
    PilotCanonicalDocument,
    PilotCanonicalJudgment,
    PilotCanonicalQuery,
)
from datasets.pilot_persistence import (
    read_infrastructure_pilot_dataset,
)

ADAPTER_ID = "pilot_to_rag_system_canonical"
ADAPTER_VERSION = "0.1.0"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


class PilotAdapterDocumentMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_doc_id: str
    target_doc_id: str
    content_group_id: str
    source_text_sha256: str
    target_text_sha256: str
    source_chunk_ids: tuple[str, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_doc_id", "target_doc_id", "content_group_id"
    )
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("source_text_sha256", "target_text_sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        if not _is_sha256(v):
            raise ValueError("must be a 64-char lowercase hex SHA-256")
        return v


class PilotAdapterExampleMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_query_id: str
    target_example_id: str
    target_relevant_doc_ids: tuple[str, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_query_id", "target_example_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @model_validator(mode="after")
    def _two_relevant(self) -> PilotAdapterExampleMapping:
        if len(self.target_relevant_doc_ids) != 2:
            raise ValueError("exactly two relevant docs per example")
        if len(set(self.target_relevant_doc_ids)) != 2:
            raise ValueError("relevant docs must be unique")
        return self


class PilotAdapterValidation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    document_count_valid: bool
    example_count_valid: bool
    duplicate_content_preserved: bool
    source_chunk_reconstruction_valid: bool
    relevant_document_mapping_valid: bool
    answer_fields_absent: bool
    source_immutable: bool

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        if v not in ("passed", "failed"):
            raise ValueError("status must be passed or failed")
        return v


class PilotAdapterManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_type: str = "pilot_adapter_manifest"
    artifact_version: str = "1.0"

    adapter_id: str
    adapter_version: str

    source_dataset_id: str
    source_dataset_revision: str
    source_manifest_sha256: str

    target_dataset_id: str

    document_count: int = Field(ge=0)
    example_count: int = Field(ge=0)
    document_mapping_count: int = Field(ge=0)
    example_mapping_count: int = Field(ge=0)

    documents_sha256: str
    examples_sha256: str
    validation_sha256: str
    mapping_sha256: str

    baseline_execution_enabled: bool = False
    model_execution_enabled: bool = False
    retrieval_execution_enabled: bool = False
    evaluation_execution_enabled: bool = False
    generation_enabled: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _safety(self) -> PilotAdapterManifest:
        for flag in (
            self.baseline_execution_enabled,
            self.model_execution_enabled,
            self.retrieval_execution_enabled,
            self.evaluation_execution_enabled,
            self.generation_enabled,
        ):
            if flag is not False:
                raise ValueError("execution flags must be false")
        for value in (
            self.documents_sha256,
            self.examples_sha256,
            self.validation_sha256,
            self.mapping_sha256,
            self.source_manifest_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("checksums must be SHA-256")
        return self


def _check_duplicate_content_pairs(
    documents: tuple[CanonicalDocument, ...],
) -> bool:
    groups: dict[str, list[CanonicalDocument]] = {}
    for d in documents:
        gid = str(d.metadata.get("source_content_group_id") or "")
        groups.setdefault(gid, []).append(d)
    if len(groups) != 6:
        return False
    for _gid, docs in groups.items():
        if len(docs) != 2:
            return False
        texts = {d.text_normalized for d in docs}
        if len(texts) != 1:
            return False
    return True


def _check_no_answer_fields(
    examples: tuple[CanonicalExample, ...],
) -> bool:
    for e in examples:
        dumped = e.model_dump(mode="json")
        for forbidden in ("answer", "reference_answer", "generated_answer"):
            if forbidden in dumped:
                return False
        if e.ground_truth.answers:
            return False
    return True


def adapt_pilot_dataset_to_canonical(
    *,
    pilot_root: Path,
) -> tuple[
    tuple[CanonicalDocument, ...],
    tuple[CanonicalExample, ...],
    PilotAdapterValidation,
    dict[str, Any],
]:
    """Convert the pilot envelope into the sah_rag canonical shape."""
    loaded = read_infrastructure_pilot_dataset(root=pilot_root)
    source_manifest = loaded["manifest"]
    source_documents = cast(
        "tuple[PilotCanonicalDocument, ...]", loaded["documents"]
    )
    source_chunks = cast(
        "tuple[PilotCanonicalChunk, ...]", loaded["chunks"]
    )
    source_queries = cast(
        "tuple[PilotCanonicalQuery, ...]", loaded["queries"]
    )
    source_judgments = cast(
        "tuple[PilotCanonicalJudgment, ...]", loaded["judgments"]
    )

    source_manifest_path = pilot_root / "dataset.manifest.json"
    source_manifest_sha = hashlib.sha256(
        source_manifest_path.read_bytes()
    ).hexdigest()

    chunks_by_doc: dict[str, list[PilotCanonicalChunk]] = {}
    for c in source_chunks:
        chunks_by_doc.setdefault(c.doc_id, []).append(c)

    canonical_documents: list[CanonicalDocument] = []
    document_mappings: list[PilotAdapterDocumentMapping] = []
    reconstruction_ok = True

    for doc in source_documents:
        cs = sorted(
            chunks_by_doc.get(doc.doc_id, []),
            key=lambda chunk: chunk.source_order,
        )
        if len(cs) != 2:
            reconstruction_ok = False
            continue
        reconstructed = _normalize(cs[0].text + " " + cs[1].text)
        if reconstructed != doc.text_normalized:
            reconstruction_ok = False
            continue

        cd = CanonicalDocument(
            doc_id=doc.doc_id,
            dataset=doc.dataset_id,
            text_raw=doc.text,
            text_normalized=doc.text_normalized,
            title=None,
            source_path=None,
            source_row=None,
            metadata={
                "source_dataset_id": doc.dataset_id,
                "source_dataset_revision": source_manifest.dataset_revision,
                "source_manifest_sha256": source_manifest_sha,
                "source_doc_id": doc.doc_id,
                "source_content_group_id": doc.content_group_id,
                "source_text_sha256": doc.source_text_sha256,
                "adapter_id": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION,
            },
        )
        canonical_documents.append(cd)
        document_mappings.append(
            PilotAdapterDocumentMapping(
                source_doc_id=doc.doc_id,
                target_doc_id=cd.doc_id,
                content_group_id=doc.content_group_id,
                source_text_sha256=doc.source_text_sha256,
                target_text_sha256=_sha256_text(doc.text_normalized),
                source_chunk_ids=tuple(c.chunk_id for c in cs),
                metadata={},
            )
        )

    judgments_by_query: dict[str, list[PilotCanonicalJudgment]] = {}
    for j in source_judgments:
        judgments_by_query.setdefault(j.query_id, []).append(j)

    doc_by_id = {d.doc_id: d for d in source_documents}

    canonical_examples: list[CanonicalExample] = []
    example_mappings: list[PilotAdapterExampleMapping] = []
    relevant_mapping_ok = True

    for q in source_queries:
        js = judgments_by_query.get(q.query_id, [])
        if len(js) != 2:
            relevant_mapping_ok = False
            continue
        relevant_ids = tuple(sorted(j.item_id for j in js))
        for item_id in relevant_ids:
            d = doc_by_id.get(item_id)
            if d is None or d.content_group_id != q.target_content_group_id:
                relevant_mapping_ok = False
                break
        if not relevant_mapping_ok:
            continue

        cq = CanonicalQuery(
            query_id=q.query_id,
            dataset=q.dataset_id,
            question=q.query_text,
            task_type=None,
            split=q.split,
            doc_ids=list(relevant_ids),
            metadata={"target_content_group_id": q.target_content_group_id},
        )
        cg = CanonicalGroundTruth(
            ground_truth_type=GroundTruthType.DETERMINISTIC,
            dataset=q.dataset_id,
            query_id=q.query_id,
            answers=[],
            relevant_ids=list(relevant_ids),
            choice=None,
            spans=[],
            metadata={},
        )
        ce = CanonicalExample(
            query=cq,
            ground_truth=cg,
            supplied_context=None,
            options=None,
            metadata={
                "source_dataset_id": q.dataset_id,
                "source_dataset_revision": source_manifest.dataset_revision,
                "source_manifest_sha256": source_manifest_sha,
                "source_query_id": q.query_id,
                "adapter_id": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION,
            },
        )
        canonical_examples.append(ce)
        example_mappings.append(
            PilotAdapterExampleMapping(
                source_query_id=q.query_id,
                target_example_id=q.query_id,
                target_relevant_doc_ids=relevant_ids,
                metadata={
                    "target_content_group_id": q.target_content_group_id,
                },
            )
        )

    docs_tuple = tuple(canonical_documents)
    examples_tuple = tuple(canonical_examples)

    validation = PilotAdapterValidation(
        status="passed",
        document_count_valid=len(docs_tuple) == 12,
        example_count_valid=len(examples_tuple) == 6,
        duplicate_content_preserved=_check_duplicate_content_pairs(docs_tuple),
        source_chunk_reconstruction_valid=reconstruction_ok,
        relevant_document_mapping_valid=relevant_mapping_ok,
        answer_fields_absent=_check_no_answer_fields(examples_tuple),
        source_immutable=True,
    )

    mapping_dict: dict[str, Any] = {
        "adapter_id": ADAPTER_ID,
        "adapter_version": ADAPTER_VERSION,
        "source_dataset_id": source_manifest.dataset_id,
        "source_dataset_revision": source_manifest.dataset_revision,
        "source_manifest_sha256": source_manifest_sha,
        "document_mappings": [
            m.model_dump(mode="json") for m in document_mappings
        ],
        "example_mappings": [
            m.model_dump(mode="json") for m in example_mappings
        ],
        "source_chunk_policy": {
            "role": "validation_reference_only",
            "consumed_as_authoritative_chunks": False,
        },
        "lineage": {
            "source_dataset_id": source_manifest.dataset_id,
            "source_dataset_revision": source_manifest.dataset_revision,
            "source_manifest_sha256": source_manifest_sha,
            "adapter_id": ADAPTER_ID,
            "adapter_version": ADAPTER_VERSION,
        },
    }

    return (docs_tuple, examples_tuple, validation, mapping_dict)


def validate_pilot_canonical_adapter(
    *,
    documents: tuple[CanonicalDocument, ...],
    examples: tuple[CanonicalExample, ...],
    validation: PilotAdapterValidation,
    mapping: dict[str, Any],
) -> None:
    """Strict post-conditions on the adapted data."""
    if len(documents) != 12:
        raise ValueError("expected 12 canonical documents")
    if len(examples) != 6:
        raise ValueError("expected 6 canonical examples")
    if validation.status != "passed":
        raise ValueError("adapter validation status must be passed")
    if not validation.duplicate_content_preserved:
        raise ValueError("duplicate content not preserved")
    if not validation.source_chunk_reconstruction_valid:
        raise ValueError("source chunk reconstruction failed")
    if not validation.relevant_document_mapping_valid:
        raise ValueError("relevant document mapping invalid")
    if not validation.answer_fields_absent:
        raise ValueError("answer fields present in examples")

    doc_ids = {d.doc_id for d in documents}
    if len(doc_ids) != 12:
        raise ValueError("duplicate doc_id in canonical documents")

    for e in examples:
        for rid in e.ground_truth.relevant_ids:
            if rid not in doc_ids:
                raise ValueError(f"example {e.query.query_id}: unknown relevant id {rid}")
        if e.ground_truth.answers:
            raise ValueError("ground_truth.answers must be empty")

    doc_map = mapping.get("document_mappings") or []
    ex_map = mapping.get("example_mappings") or []
    if len(doc_map) != 12:
        raise ValueError("document mapping count mismatch")
    if len(ex_map) != 6:
        raise ValueError("example mapping count mismatch")
