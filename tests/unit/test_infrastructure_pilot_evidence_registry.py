"""Tests for the Step 90 evidence registry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "infrastructure_pilot_evidence_registry.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_infrastructure_pilot_evidence_registry.py"

REQUIRED_STAGES = (
    "dataset_materialization",
    "canonical_adapter",
    "parser_chunker",
    "embedding",
    "qdrant",
    "retrieval",
    "evaluation",
)


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_registry():
    assert _run().returncode == 0


def test_all_required_stages_present():
    stages = _load()["stages"]
    for name in REQUIRED_STAGES:
        assert name in stages, name


def test_every_commit_is_40_char_sha():
    stages = _load()["stages"]
    for name, s in stages.items():
        sha = s["commit"]
        assert len(sha) == 40, name


def test_runtime_manifests_validate():
    assert _run().returncode == 0


def test_critical_counts_match_filesystem():
    stages = _load()["stages"]
    assert stages["embedding"]["critical_counts"]["embedding_records"] == 12
    assert stages["qdrant"]["critical_counts"]["points"] == 12
    assert stages["retrieval"]["critical_counts"]["ranked_documents"] == 72
    assert stages["evaluation"]["critical_counts"]["query_metric_rows"] == 24


def test_model_identity_frozen():
    model = _load()["stages"]["embedding"]["model"]
    assert model["model_id"] == "BAAI/bge-m3"
    assert model["model_revision"] == "5617a9f61b028005a4858fdac845db406aefb181"


def test_qdrant_collection_identity_frozen():
    collection = _load()["stages"]["qdrant"]["collection"]
    assert collection["name"] == "rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0"
    assert collection["mode"] == "local_persistence"