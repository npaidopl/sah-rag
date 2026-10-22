"""Sidecar manifests for generated RAG artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from sah_rag.artifacts.fingerprints import (
    count_nonempty_lines,
    file_size_bytes,
    sha256_file,
)
from sah_rag.artifacts.models import ArtifactBaseModel

JsonObject = dict[str, Any]


class ArtifactInputReference(ArtifactBaseModel):
    """Reference to one consumed input artifact."""

    artifact_type: str
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    record_count: int | None = Field(
        default=None,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_input_reference(
        self,
    ) -> ArtifactInputReference:
        """Validate required input identity fields."""

        required_values = (
            self.artifact_type,
            self.path,
            self.sha256,
        )

        if any(
            not value.strip()
            for value in required_values
        ):
            raise ValueError(
                "Input-artifact fields cannot be empty."
            )

        if len(self.sha256) != 64:
            raise ValueError(
                "Input SHA-256 fingerprints must "
                "contain 64 hexadecimal characters."
            )

        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "Input SHA-256 fingerprint is not hexadecimal."
            ) from exc

        return self


class ArtifactComponentReference(ArtifactBaseModel):
    """Component that produced an artifact."""

    component_type: str
    component_id: str
    component_version: str
    configuration_hash: str
    configuration: JsonObject = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_component(
        self,
    ) -> ArtifactComponentReference:
        """Require complete component identity."""

        required_values = (
            self.component_type,
            self.component_id,
            self.component_version,
            self.configuration_hash,
        )

        if any(
            not value.strip()
            for value in required_values
        ):
            raise ValueError(
                "Artifact-component fields cannot be empty."
            )

        return self


class ArtifactSidecarManifest(ArtifactBaseModel):
    """Completed sidecar manifest for one artifact file."""

    manifest_type: str = "artifact_sidecar_manifest"
    manifest_version: str = "1.0"

    artifact_type: str
    artifact_path: str
    artifact_sha256: str
    artifact_size_bytes: int = Field(ge=0)
    record_count: int = Field(ge=0)

    dataset_id: str | None = None
    experiment_id: str | None = None

    component: ArtifactComponentReference
    inputs: tuple[ArtifactInputReference, ...] = ()

    code_revision: str | None = None
    execution_profile: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    completed: bool = True

    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(
        self,
    ) -> ArtifactSidecarManifest:
        """Validate artifact identity and checksum."""

        required_values = (
            self.artifact_type,
            self.artifact_path,
            self.artifact_sha256,
        )

        if any(
            not value.strip()
            for value in required_values
        ):
            raise ValueError(
                "Artifact-manifest fields cannot be empty."
            )

        if len(self.artifact_sha256) != 64:
            raise ValueError(
                "Artifact SHA-256 fingerprint must "
                "contain 64 hexadecimal characters."
            )

        try:
            int(self.artifact_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "Artifact SHA-256 fingerprint "
                "is not hexadecimal."
            ) from exc

        if not self.completed:
            raise ValueError(
                "Published artifact manifests must "
                "represent completed artifacts."
            )

        return self


def build_input_reference(
    *,
    artifact_type: str,
    path: Path,
    record_count: int | None = None,
) -> ArtifactInputReference:
    """Create a fingerprinted reference to an input file."""

    return ArtifactInputReference(
        artifact_type=artifact_type,
        path=str(path.resolve()),
        sha256=sha256_file(path),
        size_bytes=file_size_bytes(path),
        record_count=record_count,
    )


def build_jsonl_sidecar_manifest(
    *,
    artifact_type: str,
    artifact_path: Path,
    component: ArtifactComponentReference,
    dataset_id: str | None = None,
    experiment_id: str | None = None,
    inputs: tuple[ArtifactInputReference, ...] = (),
    code_revision: str | None = None,
    execution_profile: str | None = None,
    metadata: JsonObject | None = None,
) -> ArtifactSidecarManifest:
    """Create a completed sidecar manifest for JSONL."""

    return ArtifactSidecarManifest(
        artifact_type=artifact_type,
        artifact_path=str(artifact_path.resolve()),
        artifact_sha256=sha256_file(
            artifact_path
        ),
        artifact_size_bytes=file_size_bytes(
            artifact_path
        ),
        record_count=count_nonempty_lines(
            artifact_path
        ),
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        component=component,
        inputs=inputs,
        code_revision=code_revision,
        execution_profile=execution_profile,
        completed=True,
        metadata=(
            metadata
            if metadata is not None
            else {}
        ),
    )


def build_file_sidecar_manifest(
    *,
    artifact_type: str,
    artifact_path: Path,
    record_count: int,
    component: ArtifactComponentReference,
    dataset_id: str | None = None,
    experiment_id: str | None = None,
    inputs: tuple[ArtifactInputReference, ...] = (),
    code_revision: str | None = None,
    execution_profile: str | None = None,
    metadata: JsonObject | None = None,
) -> ArtifactSidecarManifest:
    """Create a completed sidecar for a general artifact file."""

    if record_count < 0:
        raise ValueError("record_count cannot be negative.")

    return ArtifactSidecarManifest(
        artifact_type=artifact_type,
        artifact_path=str(artifact_path.resolve()),
        artifact_sha256=sha256_file(artifact_path),
        artifact_size_bytes=file_size_bytes(artifact_path),
        record_count=record_count,
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        component=component,
        inputs=inputs,
        code_revision=code_revision,
        execution_profile=execution_profile,
        completed=True,
        metadata=metadata if metadata is not None else {},
    )
