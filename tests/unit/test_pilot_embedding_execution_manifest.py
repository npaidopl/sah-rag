"""Tests for the pilot BGE-M3 execution manifest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_manifest.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_embedding_execution_manifest.py"

FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_manifest():
    r = _run()
    assert r.returncode == 0, r.stderr


def test_execution_id_and_contract_version():
    execution = _load()["execution"]
    assert execution["execution_id"] == "pilot-bge-m3-embedding-v1"
    assert execution["contract_version"] == "0.1.0"


def test_status_is_valid():
    status = _load()["execution"]["status"]
    assert status in ("PLANNED", "SUCCEEDED", "INTERRUPTED")


def test_only_embedding_execution_approved():
    approval = _load()["approval"]
    assert approval["embedding_execution_approved"] is True
    for flag in (
        "model_download_approved",
        "qdrant_execution_approved",
        "retrieval_execution_approved",
        "evaluation_execution_approved",
        "generation_execution_approved",
    ):
        assert approval[flag] is False, flag


def test_model_revision_pinned():
    model = _load()["model"]
    assert model["model_id"] == "BAAI/bge-m3"
    assert model["model_revision"] == FROZEN_REVISION
    assert model["cache_status"] == "VERIFIED"


def test_results_state_matches_status():
    manifest = _load()
    status = manifest["execution"]["status"]
    results = manifest["results"]
    if status == "PLANNED":
        assert results["model_loaded"] is False
        assert results["benchmark_performed"] is False
        assert results["embedding_materialization_performed"] is False
        assert results["dense_vector_count"] is None
        assert results["sparse_vector_count"] is None
        assert results["selected_batch_size"] is None
        assert results["output_bytes"] is None
        assert results["round_trip_valid"] is False
        assert results["checksums_valid"] is False
        assert results["deterministic_data_artifacts"] is False
    elif status == "SUCCEEDED":
        assert results["model_loaded"] is True
        assert results["benchmark_performed"] is True
        assert results["embedding_materialization_performed"] is True
        assert results["dense_vector_count"] == 12
        assert results["sparse_vector_count"] == 12
        assert results["selected_batch_size"] in (1, 2, 4)
        assert isinstance(results["output_bytes"], int) and results["output_bytes"] > 0
        assert results["round_trip_valid"] is True
        assert results["checksums_valid"] is True
        assert results["deterministic_data_artifacts"] is True
    else:
        raise AssertionError("unexpected manifest status: " + str(status))


def test_next_gate_requires_qdrant_ingestion_plan():
    next_gate = _load()["next_gate"]
    assert next_gate["required_step"] == "qdrant_ingestion_plan"
    assert next_gate["approved"] is False