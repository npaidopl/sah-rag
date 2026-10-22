"""Tests for the Step 93 architecture fidelity matrix."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "architecture_fidelity_matrix.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_architecture_fidelity_matrix.py"

ARCHS = ("sah_rag","longrag","raptor","hipporag2","lightrag","pathrag",
         "disretrieval","memorag","kag","metadata_rag","hikey")


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_matrix():
    assert _run().returncode == 0


def test_eleven_records():
    assert set(_load()["architectures"].keys()) == set(ARCHS)


def test_every_record_has_required_fields():
    for name, rec in _load()["architectures"].items():
        for f in ("implementation_status","fidelity_status",
                  "benchmark_eligibility","controlled_track",
                  "native_track","next_readiness_action"):
            assert f in rec, f"{name}: missing {f}"


def test_rag_system_native_faithful():
    a = _load()["architectures"]["sah_rag"]
    assert a["fidelity_status"] == "NATIVE_FAITHFUL"
    assert a["benchmark_eligibility"] == "BOTH_TRACKS"


def test_pathrag_paper_derived_comparator():
    a = _load()["architectures"]["pathrag"]
    assert a["fidelity_status"] == "PAPER_DERIVED_COMPARATOR"


def test_hikey_deferred():
    a = _load()["architectures"]["hikey"]
    assert a["benchmark_eligibility"] == "DEFERRED"