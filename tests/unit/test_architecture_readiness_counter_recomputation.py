"""Tests for the architecture readiness counter recomputation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "recompute_architecture_readiness_counters.py"
)
SNAPSHOT = (
    REPO_ROOT
    / "baseline_audit"
    / "architecture_portfolio_readiness_snapshot.yaml"
)


def test_recompute_produces_frozen_snapshot() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    with SNAPSHOT.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["snapshot"]["status"] == "FROZEN"
    assert data["counters"]["dataset_ready_count"] == 3
    assert data["counters"]["deferred_count"] == 1
    assert "pathrag" in data["dataset_ready_members"]
    assert "raptor" in data["deferred_members"]


def test_recompute_is_idempotent() -> None:
    r1 = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r1.returncode == 0
    with SNAPSHOT.open("r", encoding="utf-8") as f:
        s1 = yaml.safe_load(f)
    r2 = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r2.returncode == 0
    with SNAPSHOT.open("r", encoding="utf-8") as f:
        s2 = yaml.safe_load(f)
    # Counters and members must be stable across reruns.
    assert s1["counters"] == s2["counters"]
    assert s1["dataset_ready_members"] == s2["dataset_ready_members"]
    assert s1["deferred_members"] == s2["deferred_members"]


def test_dataset_ready_count_is_three() -> None:
    with SNAPSHOT.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert set(data["dataset_ready_members"]) == {
        "longrag",
        "pathrag",
        "rag_system",
    }


def test_deferred_contains_only_raptor() -> None:
    with SNAPSHOT.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["deferred_members"] == ["raptor"]


def test_counter_source_is_records() -> None:
    with SNAPSHOT.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["counter_source"]["type"] == (
        "ARCHITECTURE_READINESS_RECORDS"
    )
    assert data["counter_source"]["manual_counter_edits_allowed"] is False