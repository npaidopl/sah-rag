"""Tests for the pilot parser+chunker contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_parser_chunker_contract.yaml"
MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_parser_chunker_manifest.yaml"
VALIDATOR_C = REPO_ROOT / "scripts" / "validate_pilot_parser_chunker_contract.py"
VALIDATOR_M = REPO_ROOT / "scripts" / "validate_pilot_parser_chunker_manifest.py"


def _load(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _run(p: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(p)], cwd=REPO_ROOT,
                          capture_output=True, text=True)


def test_contract_validator_passes():
    r = _run(VALIDATOR_C)
    assert r.returncode == 0, r.stderr


def test_only_parser_and_chunker_enabled():
    ex = _load(CONTRACT)["execution"]
    assert ex["parser_enabled"] is True
    assert ex["chunker_enabled"] is True
    assert ex["embedding_enabled"] is False
    assert ex["index_enabled"] is False
    assert ex["retrieval_enabled"] is False
    assert ex["evaluation_enabled"] is False
    assert ex["generation_enabled"] is False


def test_cpu_only_no_network():
    data = _load(CONTRACT)
    lim = data["resource_limits"]
    assert lim["cpu_only"] is True
    assert lim["gpu_allowed"] is False
    assert lim["maximum_worker_count"] == 1
    ex = data["execution"]
    assert ex["network_access_allowed"] is False
    assert ex["hosted_requests_allowed"] is False
    assert ex["foreign_code_execution_allowed"] is False


def test_source_chunks_reference_only():
    rc = _load(CONTRACT)["source_reference_chunks"]
    assert rc["role"] == "validation_reference_only"
    assert rc["equality_with_derived_chunks_required"] is False
    assert rc["boundary_equality_required"] is False
    assert rc["count_equality_required"] is False


def test_chunker_settings_frozen():
    ch = _load(CONTRACT)["chunker"]
    assert ch["model_input_token_limit"] == 256
    assert ch["content_token_overlap"] == 32


def test_manifest_validator_passes():
    r = _run(VALIDATOR_M)
    assert r.returncode == 0, r.stderr


def test_manifest_approvals_are_correct():
    ap = _load(MANIFEST)["approval"]
    assert ap["parser_execution_approved"] is True
    assert ap["chunker_execution_approved"] is True
    assert ap["embedding_execution_approved"] is False
    assert ap["retrieval_execution_approved"] is False
