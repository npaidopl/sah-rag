"""Cross-baseline execution contract for Wave-1 retrieval evaluation.

Frozen policy (do not change silently later):

- Experiment cell identity is canonical SHA-256 over the frozen
  identity ingredients.
- Execution status transitions are frozen.
- Final answer generation is disabled at every stage.
- Every persisted artifact is checksummed.
- Succeeded runs are immutable.
- PathRAG comparator execution is permitted; foreign PathRAG is blocked.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sah_rag.artifacts import sha256_fingerprint

EXECUTION_STATUS_VALUES = (
    "PLANNED",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "INTERRUPTED",
    "CANCELLED",
)

TERMINAL_STATUS_VALUES = ("SUCCEEDED", "FAILED", "CANCELLED")

STATUS_TRANSITIONS: dict[str, set[str]] = {
    "PLANNED": {"RUNNING", "CANCELLED"},
    "RUNNING": {"SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED"},
    "INTERRUPTED": {"RUNNING", "CANCELLED"},
    "SUCCEEDED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}


class ExperimentCellIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_cell_id: str
    dataset_id: str
    dataset_revision: str
    split: str
    architecture_id: str
    adapter_id: str
    adapter_version: str
    architecture_configuration_hash: str
    canonical_input_manifest_sha256: str
    random_seed: int = Field(ge=0)
    code_revision: str

    @field_validator(
        "experiment_cell_id",
        "dataset_id",
        "dataset_revision",
        "split",
        "architecture_id",
        "adapter_id",
        "adapter_version",
        "architecture_configuration_hash",
        "canonical_input_manifest_sha256",
        "code_revision",
    )
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("random_seed")
    @classmethod
    def _nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("random_seed must be non-negative")
        return v


class DatasetInputEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    documents_file: str = "documents.jsonl"
    chunks_file: str = "chunks.jsonl"
    queries_file: str = "queries.jsonl"
    judgments_file: str = "judgments.jsonl"
    manifest_file: str = "dataset.manifest.json"


class ExecutionFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_type: str
    failure_stage: str
    message: str
    retryable: bool
    traceback_sha256: str | None = None
    occurred_at_utc: datetime


class ExecutionResourceUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_validation_seconds: float = 0.0
    preprocessing_seconds: float = 0.0
    indexing_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    persistence_seconds: float = 0.0
    total_execution_seconds: float = 0.0

    peak_process_ram_bytes: int = 0
    peak_gpu_memory_bytes: int | None = None

    input_bytes: int = 0
    index_bytes: int = 0
    retrieval_artifact_bytes: int = 0
    total_output_bytes: int = 0

    embedding_request_count: int = 0
    embedding_input_tokens: int = 0
    llm_request_count: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_failure_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0

    document_count: int = 0
    chunk_count: int = 0
    query_count: int = 0
    judgment_count: int = 0
    retrieved_item_count: int = 0
    expanded_document_count: int = 0

    @field_validator(
        "input_validation_seconds",
        "preprocessing_seconds",
        "indexing_seconds",
        "retrieval_seconds",
        "evaluation_seconds",
        "persistence_seconds",
        "total_execution_seconds",
    )
    @classmethod
    def _finite_nonneg_duration(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0.0:
            raise ValueError("duration must be finite and non-negative")
        return v

    @field_validator(
        "peak_process_ram_bytes",
        "input_bytes",
        "index_bytes",
        "retrieval_artifact_bytes",
        "total_output_bytes",
        "embedding_request_count",
        "embedding_input_tokens",
        "llm_request_count",
        "llm_input_tokens",
        "llm_output_tokens",
        "llm_failure_count",
        "cache_hit_count",
        "cache_miss_count",
        "document_count",
        "chunk_count",
        "query_count",
        "judgment_count",
        "retrieved_item_count",
        "expanded_document_count",
    )
    @classmethod
    def _nonneg_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v


class ExecutionArtifactRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_type: str
    relative_path: str
    sha256: str
    bytes: int = Field(ge=0)

    @field_validator("artifact_type", "relative_path", "sha256")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("relative_path")
    @classmethod
    def _no_absolute_or_traversal(cls, v: str) -> str:
        if v.startswith("/") or v.startswith("\\") or ".." in v.split("/"):
            raise ValueError("relative_path must stay inside the run root")
        return v


class ArchitectureExecutionManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_cell_id: str
    architecture_id: str
    architecture_display_name: str
    architecture_classification: str
    implementation_class: str
    adapter_id: str
    adapter_version: str
    architecture_source_revision: str
    project_code_revision: str
    dataset_id: str
    dataset_revision: str
    split: str
    random_seed: int = Field(ge=0)
    configuration_hash: str
    canonical_input_manifest_sha256: str
    execution_environment_id: str
    execution_status: str
    started_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None
    failure: ExecutionFailure | None = None
    resource_usage: ExecutionResourceUsage = Field(
        default_factory=ExecutionResourceUsage
    )
    artifact_inventory: tuple[ExecutionArtifactRecord, ...] = ()
    safety: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    final_generation_enabled: bool = False
    answer_evaluation_enabled: bool = False
    qa_model_enabled: bool = False
    llm_as_judge_enabled: bool = False

    @field_validator("execution_status")
    @classmethod
    def _valid_status(cls, v: str) -> str:
        if v not in EXECUTION_STATUS_VALUES:
            raise ValueError(f"invalid execution_status: {v}")
        return v

    @field_validator(
        "final_generation_enabled",
        "answer_evaluation_enabled",
        "qa_model_enabled",
        "llm_as_judge_enabled",
    )
    @classmethod
    def _must_be_false(cls, v: bool) -> bool:
        if v is not False:
            raise ValueError("generation-free retrieval boundary violated")
        return v

    @model_validator(mode="after")
    def _invariants(self) -> ArchitectureExecutionManifest:
        if self.execution_status == "SUCCEEDED":
            if self.completed_at_utc is None:
                raise ValueError("SUCCEEDED requires completed_at_utc")
            if self.failure is not None:
                raise ValueError("SUCCEEDED must not carry a failure record")
            if not self.artifact_inventory:
                raise ValueError("SUCCEEDED requires artifact_inventory")
        if self.execution_status == "FAILED" and self.failure is None:
            raise ValueError("FAILED requires a failure record")
        if (
            self.execution_status in ("PLANNED", "RUNNING")
            and self.completed_at_utc is not None
        ):
            raise ValueError(
                f"{self.execution_status} must not have completed_at_utc"
            )
        if (
            self.completed_at_utc is not None
            and self.started_at_utc is not None
            and self.completed_at_utc < self.started_at_utc
        ):
            raise ValueError("completed_at_utc precedes started_at_utc")
        return self


class ReadinessRegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture_id: str
    readiness_status: str
    readiness_declaration: str
    execution_label: str
    final_generation_enabled: bool = False
    foreign_repository_status: str | None = None
    implementation_class: str | None = None

    @field_validator("architecture_id", "readiness_declaration", "execution_label")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("readiness_status")
    @classmethod
    def _status(cls, v: str) -> str:
        if v != "READY_FOR_DATA":
            raise ValueError("readiness_status must be READY_FOR_DATA")
        return v

    @field_validator("final_generation_enabled")
    @classmethod
    def _no_gen(cls, v: bool) -> bool:
        if v is not False:
            raise ValueError("final_generation_enabled must be False")
        return v


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_experiment_cell_id(
    *,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    architecture_id: str,
    adapter_id: str,
    adapter_version: str,
    architecture_configuration_hash: str,
    canonical_input_manifest_sha256: str,
    random_seed: int,
    code_revision: str,
) -> str:
    """Deterministic experiment-cell identity."""
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    payload = {
        "kind": "experiment_cell_identity",
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split": split,
        "architecture_id": architecture_id,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "architecture_configuration_hash": architecture_configuration_hash,
        "canonical_input_manifest_sha256": canonical_input_manifest_sha256,
        "random_seed": random_seed,
        "code_revision": code_revision,
    }
    fp = sha256_fingerprint(payload)
    return f"experiment-cell-{fp}"


def validate_execution_status_transition(
    *,
    previous_status: str,
    next_status: str,
) -> None:
    if previous_status not in STATUS_TRANSITIONS:
        raise ValueError(f"unknown previous status: {previous_status}")
    if next_status not in EXECUTION_STATUS_VALUES:
        raise ValueError(f"unknown next status: {next_status}")
    if next_status not in STATUS_TRANSITIONS[previous_status]:
        raise ValueError(
            f"transition {previous_status} -> {next_status} not allowed"
        )


def transition_execution_manifest(
    *,
    manifest: ArchitectureExecutionManifest,
    next_status: str,
    completed_at_utc: datetime | None = None,
    failure: ExecutionFailure | None = None,
) -> ArchitectureExecutionManifest:
    validate_execution_status_transition(
        previous_status=manifest.execution_status,
        next_status=next_status,
    )
    update: dict[str, Any] = {"execution_status": next_status}
    if completed_at_utc is not None:
        update["completed_at_utc"] = completed_at_utc
    if failure is not None:
        update["failure"] = failure
    return manifest.model_copy(update=update)
