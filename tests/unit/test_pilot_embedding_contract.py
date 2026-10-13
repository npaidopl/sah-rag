"""Tests for the pilot embedding contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_embedding_contract.py"

FROZEN = "5617a9f61b028005a4858fdac845db406aefb181"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(VALIDATOR)],
                          cwd=REPO_ROOT, capture_output=True, text=True)


def test_validator_accepts_contract():
    r = _run()
    assert r.returncode == 0, r.stderr


def test_frozen_model_id():
    assert _load()["model"]["model_id"] == "BAAI/bge-m3"


def test_frozen_revision_is_full_sha():
    rev = _load()["model"]["model_revision"]
    assert len(rev) == 40
    assert rev == FROZEN


def test_license_is_mit_from_local_evidence():
    lic = _load()["model"]["license"]
    assert lic["spdx"] == "MIT"
    assert lic["license_review_status"] == "VERIFIED_FROM_LOCAL_SNAPSHOT"


def test_dense_dimension_and_normalization():
    m = _load()["model"]
    assert m["dense_dimension"] == 1024
    assert m["dense_normalized"] is True


def test_sparse_enabled_multivector_disabled():
    caps = _load()["model"]["pilot_capabilities_enabled"]
    assert caps["dense"] is True
    assert caps["learned_sparse"] is True
    assert caps["multi_vector"] is False
