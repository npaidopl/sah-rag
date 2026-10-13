"""Tests for the Step 93 benchmark execution tracks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "benchmark_execution_tracks.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_benchmark_execution_tracks.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_tracks():
    assert _run().returncode == 0


def test_three_tracks_defined():
    t = _load()["tracks"]
    assert "track_a_controlled_architectural" in t
    assert "track_b_native_fidelity" in t
    assert "track_c_deferred_appendix" in t


def test_track_a_is_primary_and_leaderboard_eligible():
    a = _load()["tracks"]["track_a_controlled_architectural"]
    assert a["role"] == "PRIMARY"
    assert a["leaderboard_eligible"] is True


def test_track_b_is_not_leaderboard_eligible():
    b = _load()["tracks"]["track_b_native_fidelity"]
    assert b["role"] == "SECONDARY"
    assert b["leaderboard_eligible"] is False
    assert b["merge_with_controlled_scores"] is False


def test_track_b_currently_eligible_vs_potential():
    b = _load()["tracks"]["track_b_native_fidelity"]
    assert b["currently_eligible"] == ["rag_system"]
    assert b["currently_eligible_count"] == 1
    assert b["potential_candidate_count"] == 4
    assert set(b["currently_eligible"]) & set(b["potential_candidates"]) == set()


def test_track_c_is_numeric_appendix():
    c = _load()["tracks"]["track_c_deferred_appendix"]
    assert c["numeric_leaderboard_eligible"] is False


def test_tracks_not_mixed():
    assert _load()["reporting_rules"]["mixing_tracks_in_single_leaderboard"] is False