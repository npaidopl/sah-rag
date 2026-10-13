"""Tests for the architecture portfolio terminal dispositions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO = REPO_ROOT / "baseline_audit" / "architecture_portfolio_dispositions.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_architecture_portfolio_dispositions.py"

EXPECTED = {
    "longrag", "raptor", "hipporag2", "lightrag", "pathrag",
    "disretrieval", "memorag", "kag",
    "metadata_rag", "hikey",
}
READY = {"longrag", "raptor", "hipporag2", "lightrag", "pathrag"}
CONDITIONAL = {"disretrieval", "memorag", "metadata_rag"}
DEFERRED = {"kag", "hikey"}

ALLOWED = {
    "READY_FOR_DATA", "CONDITIONAL_GO",
    "DEFERRED_WITH_REASON", "BLOCKED_WITH_REASON",
}
PROHIBITED = {
    "NOT_STARTED", "UNKNOWN", "PENDING",
    "PENDING_UNBOUNDED", "UNDECIDED",
}


def _load() -> dict:
    return yaml.safe_load(PORTFOLIO.read_text(encoding="utf-8"))


def _run_validator() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_current_registry():
    result = _run_validator()
    assert result.returncode == 0, result.stderr
    assert "[OK] Architecture portfolio terminal dispositions" in result.stdout


def test_exactly_ten_architecture_slots():
    data = _load()
    assert set(data["architectures"]) == EXPECTED
    assert len(data["architectures"]) == 10


def test_all_slots_have_allowed_terminal_status():
    data = _load()
    for arch_id, entry in data["architectures"].items():
        assert entry["operational_status"] in ALLOWED, arch_id


def test_no_prohibited_status_present():
    data = _load()
    for arch_id, entry in data["architectures"].items():
        assert entry["operational_status"] not in PROHIBITED, arch_id


def test_ready_set_is_exactly_wave_1():
    data = _load()
    ready = {
        aid for aid, e in data["architectures"].items()
        if e["operational_status"] == "READY_FOR_DATA"
    }
    assert ready == READY


def test_conditional_set_is_correct():
    data = _load()
    conditional = {
        aid for aid, e in data["architectures"].items()
        if e["operational_status"] == "CONDITIONAL_GO"
    }
    assert conditional == CONDITIONAL


def test_deferred_set_is_correct():
    data = _load()
    deferred = {
        aid for aid, e in data["architectures"].items()
        if e["operational_status"] == "DEFERRED_WITH_REASON"
    }
    assert deferred == DEFERRED


def test_pathrag_foreign_execution_remains_blocked():
    data = _load()
    assert data["architectures"]["pathrag"]["foreign_execution_status"] == "BLOCKED"


def test_hikey_requires_contract_0_2_0():
    data = _load()
    assert "0.2.0" in data["architectures"]["hikey"]["next_action"]


def test_publishable_execution_remains_disabled():
    data = _load()
    assert data["execution_gate"]["publishable_execution_allowed"] is False
    assert data["portfolio"]["publishable_execution_allowed"] is False


def test_infrastructure_only_pilot_remains_allowed():
    data = _load()
    assert data["portfolio"]["infrastructure_only_pilot_allowed"] is True


def test_summary_counts_sum_to_ten():
    data = _load()
    s = data["summary"]
    assert s["total_count"] == 10
    assert (
        s["ready_for_data_count"]
        + s["conditional_go_count"]
        + s["deferred_with_reason_count"]
        + s["blocked_with_reason_count"]
        == 10
    )
