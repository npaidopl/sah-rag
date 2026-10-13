"""Tests for the Step 93 component substitution policy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "component_substitution_policy.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_component_substitution_policy.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_substitution_policy():
    assert _run().returncode == 0


def test_three_component_roles():
    r = _load()["component_roles"]
    for role in ("ARCHITECTURE_DEFINING","ARCHITECTURE_SUPPORTING","OPTIONAL_DOWNSTREAM"):
        assert role in r


def test_llm_final_generation_disabled():
    assert _load()["llm_policy"]["final_generation"]["enabled"] is False


def test_llm_stages_preserved_when_architecture_defining():
    llm = _load()["llm_policy"]
    assert llm["index_time"]["preserve_when_architecture_defining"] is True
    assert llm["query_time"]["preserve_when_architecture_defining"] is True


def test_storage_substitution_policy_defined():
    st = _load()["storage_substitution_policy"]
    assert len(st["common_qdrant_allowed_when"]) >= 1
    assert len(st["native_storage_required_when"]) >= 1


def test_equivalence_test_requires_top_k_overlap():
    t = _load()["required_substitution_test"]
    assert "top_10_overlap" in t["compare"]
    assert "top_100_overlap" in t["compare"]