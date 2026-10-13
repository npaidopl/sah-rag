"""Tests for the pilot Qdrant ingestion planning manifest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_qdrant_ingestion_manifest.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_qdrant_ingestion_manifest.py"


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


def test_execution_id():
    assert _load()["execution"]["execution_id"] == "pilot-qdrant-ingestion-v1"


def test_approvals_frozen():
    approval = _load()["approval"]
    assert approval["environment_inspection_approved"] is True
    assert approval["ingestion_planning_approved"] is True
    for flag in (
        "collection_creation_approved",
        "vector_ingestion_approved",
        "retrieval_execution_approved",
        "evaluation_execution_approved",
        "generation_execution_approved",
    ):
        assert approval[flag] is False, flag


def test_plan_expected_counts():
    plan = _load()["plan"]
    assert plan["embedding_record_count"] == 12
    assert plan["expected_point_count"] == 12
    assert plan["dense_vector_count"] == 12
    assert plan["sparse_vector_count"] == 12


def test_results_all_false_before_execution():
    results = _load()["results"]
    for flag in (
        "qdrant_client_opened",
        "storage_root_created",
        "collection_created",
        "vectors_ingested",
        "round_trip_valid",
        "checksums_valid",
    ):
        assert results[flag] is False, flag
    assert results["point_count"] is None
    assert results["output_bytes"] is None


def test_next_gate_requires_bounded_ingestion():
    next_gate = _load()["next_gate"]
    assert next_gate["required_step"] == "bounded_qdrant_ingestion"
    assert next_gate["approved"] is False