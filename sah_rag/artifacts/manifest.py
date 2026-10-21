"""Experiment manifest models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, model_validator

from rag_system.artifacts.models import (
    ArtifactBaseModel,
    JsonObject,
)


class ExperimentStatus(StrEnum):
    """Lifecycle states of an experiment."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ExecutionEnvironment(StrEnum):
    """Supported experiment execution environments."""

    LOCAL = "local"
    COLAB = "colab"
    CLOUD = "cloud"
    UNKNOWN = "unknown"


class ComponentReference(ArtifactBaseModel):
    """Versioned reference to one pipeline component."""

    component_id: str
    version: str
    config_hash: str
    configuration: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference(self) -> ComponentReference:
        """Require non-empty component identity fields."""

        values = (
            self.component_id,
            self.version,
            self.config_hash,
        )

        if any(not value.strip() for value in values):
            raise ValueError(
                "Component reference fields cannot be empty."
            )

        return self


class ExecutionReference(ArtifactBaseModel):
    """Recorded execution environment for one experiment."""

    profile_id: str
    environment: ExecutionEnvironment
    device: str
    precision: str
    python_version: str

    operating_system: str | None = None
    cpu: str | None = None
    gpu: str | None = None
    ram_gb: float | None = Field(default=None, gt=0)
    vram_gb: float | None = Field(default=None, gt=0)

    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_execution(self) -> ExecutionReference:
        """Require core execution fields."""

        values = (
            self.profile_id,
            self.device,
            self.precision,
            self.python_version,
        )

        if any(not value.strip() for value in values):
            raise ValueError(
                "Execution reference fields cannot be empty."
            )

        return self


class ArtifactReference(ArtifactBaseModel):
    """Reference to a generated or consumed artifact."""

    artifact_type: str
    path: str
    fingerprint: str | None = None
    record_count: int | None = Field(default=None, ge=0)
    size_bytes: int | None = Field(default=None, ge=0)
    completed: bool = False
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_artifact_reference(
        self,
    ) -> ArtifactReference:
        """Require non-empty artifact type and path."""

        if (
            not self.artifact_type.strip()
            or not self.path.strip()
        ):
            raise ValueError(
                "Artifact type and path cannot be empty."
            )

        return self


class ExperimentManifest(ArtifactBaseModel):
    """Complete identity and provenance record for one experiment."""

    artifact_type: str = "experiment_manifest"
    artifact_version: str = "1.0"

    experiment_id: str
    architecture_id: str
    architecture_version: str

    dataset_id: str
    canonical_root_fingerprint: str | None = None
    canonical_manifest_fingerprint: str | None = None

    parser: ComponentReference
    chunker: ComponentReference
    embedder: ComponentReference | None = None
    vectorstore: ComponentReference | None = None
    retriever: ComponentReference | None = None
    fusion: ComponentReference | None = None
    reranker: ComponentReference | None = None
    evaluator: ComponentReference | None = None

    execution: ExecutionReference

    random_seed: int = Field(ge=0)
    code_revision: str | None = None
    configuration_hash: str

    status: ExperimentStatus = ExperimentStatus.CREATED

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None

    inputs: tuple[ArtifactReference, ...] = ()
    outputs: tuple[ArtifactReference, ...] = ()

    metrics: JsonObject = Field(default_factory=dict)
    errors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> ExperimentManifest:
        """Validate experiment identity and lifecycle timestamps."""

        required_values = (
            self.experiment_id,
            self.architecture_id,
            self.architecture_version,
            self.dataset_id,
            self.configuration_hash,
        )

        if any(not value.strip() for value in required_values):
            raise ValueError(
                "Experiment identity fields cannot be empty."
            )

        if (
            self.completed_at is not None
            and self.started_at is None
        ):
            raise ValueError(
                "completed_at requires started_at."
            )

        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError(
                "completed_at cannot precede started_at."
            )

        if (
            self.status is ExperimentStatus.COMPLETED
            and self.completed_at is None
        ):
            raise ValueError(
                "Completed experiments require completed_at."
            )

        if (
            self.status is ExperimentStatus.RUNNING
            and self.started_at is None
        ):
            raise ValueError(
                "Running experiments require started_at."
            )

        return self