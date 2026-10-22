"""Tests for the sah_rag end-to-end static inspection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "baseline_audit" / "sah_rag_end_to_end_inspection.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_sah_rag_end_to_end_inspection.py"


def _load() -> dict:
    return yaml.safe_load(AUDIT.read_text(encoding="utf-8"))


def _run_validator() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_current_audit():
    r = _run_validator()
    assert r.returncode == 0, r.stderr
    assert "[OK] sah_rag end-to-end static inspection" in r.stdout


def test_pipeline_is_project_owned():
    data = _load()
    assert data["pipeline"]["ownership"] == "PROJECT_OWNED"


def test_stages_are_ordered_and_unique():
    data = _load()
    for section in ("offline", "online"):
        stages = data["stages"][section]
        ids = [s["stage_id"] for s in stages]
        assert len(ids) == len(set(ids)), section


def test_embedding_identity_is_explicit():
    data = _load()
    emb = data["embedding"]
    assert emb["dense_model_id"] == "BAAI/bge-m3"
    assert emb["dense_dimension"] == 1024
    assert emb["local_or_hosted"] == "local"


def test_storage_backend_is_explicit():
    data = _load()
    assert data["storage"]["backend"] == "Qdrant"


def test_retrieval_boundary_is_explicit():
    data = _load()
    rb = data["retrieval"]
    assert rb["score_direction"] == "higher_is_better"
    assert rb["canonical_doc_ids_exposed"] is True


def test_generation_disabled_for_pilot():
    data = _load()
    assert data["generation_boundary"]["final_generation_enabled_for_pilot"] is False


def test_canonical_compatibility_is_explicit():
    data = _load()
    cc = data["canonical_compatibility"]
    assert cc["status"] in ("DIRECT", "ADAPTER_REQUIRED", "PARTIAL", "INCOMPATIBLE")


def test_host_feasibility_is_explicit():
    data = _load()
    hf = data["host_feasibility"]
    assert hf["current_host_status"] in (
        "EXECUTABLE_ON_CURRENT_HOST",
        "EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS",
        "REQUIRES_REMOTE_GPU",
        "NOT_EXECUTABLE",
    )


def test_pilot_recommendation_has_required_changes_and_prohibited_stages():
    data = _load()
    pr = data["pilot_recommendation"]
    assert pr.get("required_changes")
    assert pr.get("prohibited_stages")
