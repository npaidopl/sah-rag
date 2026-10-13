"""Tests for the Metadata-RAG paper-to-spec audit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "baseline_audit" / "metadata_rag_paper_spec_audit.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_metadata_rag_paper_spec_audit.py"


def _load() -> dict:
    return yaml.safe_load(AUDIT.read_text(encoding="utf-8"))


def _run_validator() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_current_audit():
    result = _run_validator()
    assert result.returncode == 0, result.stderr
    assert "[OK] Metadata-RAG paper-to-spec audit" in result.stdout


def test_architecture_id_and_wave_are_frozen():
    data = _load()
    assert data["architecture"]["id"] == "metadata_rag"
    assert data["architecture"]["wave"] == 3


def test_no_invented_foreign_repository():
    data = _load()
    fs = data["foreign_source"]
    assert fs["repository_registered"] is False
    assert fs["checkout_present"] is False
    assert fs["foreign_source_execution_applicable"] is False
    assert fs["repository_url"] is None


def test_paper_derived_class_requires_resolved_paper():
    data = _load()
    cls = data["classification"]
    identity = data["identity"]
    if cls["implementation_class"] == "PAPER_DERIVED_REIMPLEMENTATION":
        assert identity["status"] == "RESOLVED"


def test_leakage_policy_remains_prohibited():
    data = _load()
    lp = data["leakage_policy"]
    for key in (
        "judgments_as_metadata",
        "reference_answers_as_metadata",
        "evaluation_labels_as_metadata",
        "generated_answers_as_metadata",
        "post_retrieval_annotations_as_metadata",
    ):
        assert lp[key] == "prohibited", key


def test_retrieval_boundary_status_is_explicit():
    data = _load()
    status = data["retrieval_boundary"]["status"]
    assert status in ("PASS", "PARTIAL", "FAIL", "DESIGN_REQUIRED")


def test_traceability_status_is_explicit():
    data = _load()
    status = data["traceability"]["status"]
    assert status in (
        "NATIVE_WITH_CANONICAL_MAPPING",
        "REQUIRES_INDEPENDENT_DESIGN",
        "NOT_RECOVERABLE",
        "UNRESOLVED",
    )


def test_ready_for_data_not_permitted_by_audit():
    data = _load()
    status = data["terminal_disposition"]["status"]
    assert status != "READY_FOR_DATA"
    assert status in ("CONDITIONAL_GO", "BLOCKED_WITH_REASON", "DEFERRED_WITH_REASON")


def test_terminal_disposition_has_reason_and_next_action():
    data = _load()
    disp = data["terminal_disposition"]
    assert disp.get("reason")
    assert disp.get("next_action")
