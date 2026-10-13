"""Tests for the cross-baseline execution contract models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag_system.execution_contracts import (
    ArchitectureExecutionManifest,
    ExecutionArtifactRecord,
    ExecutionResourceUsage,
    build_experiment_cell_id,
    transition_execution_manifest,
    validate_execution_status_transition,
)


def _cell_id(**overrides) -> str:
    kwargs = dict(
        dataset_id="clerk",
        dataset_revision="rev-1",
        split="test",
        architecture_id="longrag",
        adapter_id="longrag-adapter",
        adapter_version="0.1.0",
        architecture_configuration_hash="cfg-hash",
        canonical_input_manifest_sha256="a" * 64,
        random_seed=42,
        code_revision="code-rev",
    )
    kwargs.update(overrides)
    return build_experiment_cell_id(**kwargs)


def _base_manifest(**overrides) -> ArchitectureExecutionManifest:
    kwargs: dict = dict(
        experiment_cell_id="experiment-cell-" + "a" * 64,
        architecture_id="longrag",
        architecture_display_name="LongRAG",
        architecture_classification="CONTROLLED_ADAPTATION",
        implementation_class="CONTROLLED_ADAPTATION",
        adapter_id="longrag-adapter",
        adapter_version="0.1.0",
        architecture_source_revision="ba92f95",
        project_code_revision="code-rev",
        dataset_id="clerk",
        dataset_revision="rev-1",
        split="test",
        random_seed=42,
        configuration_hash="cfg-hash",
        canonical_input_manifest_sha256="a" * 64,
        execution_environment_id="local-windows-cpu",
        execution_status="PLANNED",
        started_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )
    kwargs.update(overrides)
    return ArchitectureExecutionManifest(**kwargs)


def _artifact() -> ExecutionArtifactRecord:
    return ExecutionArtifactRecord(
        artifact_type="ranked_items",
        relative_path="retrieval/ranked_items.jsonl",
        sha256="b" * 64,
        bytes=128,
    )


def test_deterministic_cell_id():
    assert _cell_id() == _cell_id()


def test_dataset_revision_changes_cell_id():
    assert _cell_id() != _cell_id(dataset_revision="rev-2")


def test_seed_changes_cell_id():
    assert _cell_id() != _cell_id(random_seed=43)


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        _cell_id(random_seed=-1)


def test_transition_planned_to_running_allowed():
    validate_execution_status_transition(
        previous_status="PLANNED", next_status="RUNNING"
    )


def test_transition_running_to_succeeded_allowed():
    validate_execution_status_transition(
        previous_status="RUNNING", next_status="SUCCEEDED"
    )


def test_transition_running_to_interrupted_allowed():
    validate_execution_status_transition(
        previous_status="RUNNING", next_status="INTERRUPTED"
    )


def test_transition_interrupted_to_running_allowed():
    validate_execution_status_transition(
        previous_status="INTERRUPTED", next_status="RUNNING"
    )


def test_transition_succeeded_to_running_rejected():
    with pytest.raises(ValueError):
        validate_execution_status_transition(
            previous_status="SUCCEEDED", next_status="RUNNING"
        )


def test_succeeded_manifest_requires_artifacts():
    with pytest.raises(ValueError):
        _base_manifest(
            execution_status="SUCCEEDED",
            completed_at_utc=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
            artifact_inventory=(),
        )


def test_failed_manifest_requires_failure():
    with pytest.raises(ValueError):
        _base_manifest(
            execution_status="FAILED",
            completed_at_utc=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        )


def test_final_generation_flag_rejected():
    with pytest.raises(ValueError):
        _base_manifest(final_generation_enabled=True)


def test_transition_helper_returns_new_manifest():
    manifest = _base_manifest()
    transitioned = transition_execution_manifest(
        manifest=manifest, next_status="RUNNING"
    )
    assert transitioned.execution_status == "RUNNING"
    assert manifest.execution_status == "PLANNED"


def test_resource_usage_rejects_negative_counts():
    with pytest.raises(ValueError):
        ExecutionResourceUsage(document_count=-1)
