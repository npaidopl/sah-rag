"""Tests for the Step 94 architecture readiness gap registry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "architecture_readiness_gap_registry.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_architecture_readiness_gap_registry.py"

ARCHS = ("sah_rag","longrag","raptor","hipporag2","lightrag","pathrag",
         "disretrieval","memorag","kag","metadata_rag","hikey")

REQUIRED = (
    "architecture_id","implementation_status","fidelity_class",
    "controlled_track_status","native_track_status","readiness_gate",
    "current_artifacts","verified_components","missing_components",
    "architecture_defining_components","controlled_substitutions",
    "llm_dependencies","storage_policy","canonical_input_status",
    "retrieval_output_status","synthetic_acceptance_status",
    "dataset_readiness_status","blockers","next_action_class",
    "next_action","recommended_order",
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


def test_eleven_records():
    assert set(_load()["architectures"].keys()) == set(ARCHS)


def test_every_record_has_required_fields():
    a = _load()["architectures"]
    for name in ARCHS:
        for f in REQUIRED:
            assert f in a[name], f"{name}: missing {f}"


def test_every_record_has_blockers_field():
    a = _load()["architectures"]
    for name in ARCHS:
        assert a[name]["blockers"] is not None, name


def test_every_record_has_next_action():
    a = _load()["architectures"]
    for name in ARCHS:
        assert isinstance(a[name]["next_action"], str)
        assert a[name]["next_action"].strip(), name


def test_rag_system_is_dataset_ready():
    assert _load()["architectures"]["sah_rag"]["implementation_status"] == "DATASET_READY"


def test_summary_counts_match_records():
    a = _load()["architectures"]
    s = _load()["summary"]
    dr = sum(1 for r in a.values() if r["implementation_status"] == "DATASET_READY")
    sv = sum(1 for r in a.values() if r["implementation_status"] == "SYNTHETIC_VALIDATED")
    pa = sum(1 for r in a.values() if r["implementation_status"] == "PARTIAL")
    ns = sum(1 for r in a.values() if r["implementation_status"] == "NOT_STARTED")
    assert s["dataset_ready_count"] == dr
    assert s["synthetic_validated_count"] == sv
    assert s["partial_count"] == pa
    assert s["not_started_count"] == ns


def test_native_track_counts_match():
    s = _load()["summary"]
    assert s["native_track_current_eligible_count"] == 1
    assert s["native_track_potential_candidate_count"] == 4