"""Tests for experiment manifests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from rag_system.artifacts.manifest import (
    ComponentReference,
    ExecutionEnvironment,
    ExecutionReference,
    ExperimentManifest,
    ExperimentStatus,
)


def make_component(
    component_id: str,
) -> ComponentReference:
    """Create a valid versioned component reference."""
    return ComponentReference(
        component_id=component_id,
        version="0.1.0",
        config_hash=f"{component_id}-hash",
        configuration={},
    )


def make_execution() -> ExecutionReference:
    """Create a valid local execution reference."""
    return ExecutionReference(
        profile_id="local_cpu",
        environment=ExecutionEnvironment.LOCAL,
        device="cpu",
        precision="float32",
        python_version="3.12.10",
        operating_system="Windows",
        cpu="AMD Ryzen 5 5600G",
        gpu=None,
        ram_gb=16.37,
        vram_gb=None,
        metadata={},
    )


def make_manifest() -> ExperimentManifest:
    """Create a valid created experiment manifest."""
    return ExperimentManifest(
        experiment_id="proposed-rag__cuad__abc123",
        architecture_id="proposed-rag",
        architecture_version="0.1.0",
        dataset_id="cuad",
        parser=make_component("parser"),
        chunker=make_component("chunker"),
        embedder=make_component("bge-m3"),
        vectorstore=make_component("qdrant"),
        retriever=make_component("hybrid-retriever"),
        fusion=make_component("rrf"),
        reranker=None,
        evaluator=make_component("retrieval-evaluator"),
        execution=make_execution(),
        random_seed=42,
        code_revision=None,
        configuration_hash="configuration-hash",
        status=ExperimentStatus.CREATED,
    )


def test_created_manifest_is_valid() -> None:
    """A newly created manifest should validate."""
    manifest = make_manifest()
    assert manifest.dataset_id == "cuad"
    assert manifest.status is ExperimentStatus.CREATED


def test_running_manifest_requires_started_at() -> None:
    """Running state must record its start time."""
    raw_manifest = make_manifest().model_dump()
    raw_manifest["status"] = "running"
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw_manifest)


def test_completed_manifest_requires_completed_at() -> None:
    """Completed state must record completion time."""
    raw_manifest = make_manifest().model_dump()
    raw_manifest["status"] = "completed"
    raw_manifest["started_at"] = datetime.now(UTC)
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw_manifest)


def test_completed_at_cannot_precede_started_at() -> None:
    """Lifecycle timestamps must remain chronological."""
    started_at = datetime.now(UTC)
    completed_at = started_at - timedelta(seconds=1)
    raw_manifest = make_manifest().model_dump()
    raw_manifest["status"] = "completed"
    raw_manifest["started_at"] = started_at
    raw_manifest["completed_at"] = completed_at
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw_manifest)


def test_completed_manifest_is_valid() -> None:
    """A completed experiment with timestamps should validate."""
    started_at = datetime.now(UTC)
    completed_at = started_at + timedelta(seconds=2)
    raw_manifest = make_manifest().model_dump()
    raw_manifest["status"] = "completed"
    raw_manifest["started_at"] = started_at
    raw_manifest["completed_at"] = completed_at
    manifest = ExperimentManifest.model_validate(raw_manifest)
    assert manifest.status is ExperimentStatus.COMPLETED