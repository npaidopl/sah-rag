"""Tests for the Step 94 architecture readiness gap contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "architecture_readiness_gap_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_architecture_readiness_gap_contract.py"

ARCHS = ("rag_system","longrag","raptor","hipporag2","lightrag","pathrag",
         "disretrieval","memorag","kag","metadata_rag","hikey")


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_contract():
    assert _run().returncode == 0


def test_exact_eleven_architectures():
    assert tuple(_load()["architectures"]["expected"]) == ARCHS
    assert _load()["architectures"]["expected_count"] == 11


def test_static_only_scope():
    sa = _load()["static_analysis"]
    assert sa["import_project_modules"] is False
    assert sa["execute_architecture"] is False


def test_required_record_fields_complete():
    req = _load()["required_record_fields"]
    for f in ("architecture_id","implementation_status","fidelity_class",
              "controlled_track_status","native_track_status","readiness_gate",
              "blockers","next_action_class","next_action","recommended_order"):
        assert f in req, f


def test_execution_prohibitions():
    proh = _load()["prohibited"]
    for p in ("execute_architecture","load_model","call_llm","open_qdrant"):
        assert p in proh, p


def test_claims_boundary():
    cl = _load()["claims"]
    assert cl["readiness_registry_is_execution_result"] is False
    assert cl["static_gap_analysis_is_synthetic_proof"] is False
    assert cl["synthetic_proof_is_real_dataset_readiness"] is False