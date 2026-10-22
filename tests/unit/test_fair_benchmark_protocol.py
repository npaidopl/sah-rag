"""Tests for the Step 93 fair benchmark protocol."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "fair_benchmark_protocol.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_fair_benchmark_protocol.py"

ARCHS = ("sah_rag","longrag","raptor","hipporag2","lightrag","pathrag",
         "disretrieval","memorag","kag","metadata_rag","hikey")
DATASETS = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_protocol():
    assert _run().returncode == 0


def test_exactly_eleven_architectures():
    assert tuple(_load()["architectures"]["expected"]) == ARCHS
    assert _load()["architectures"]["expected_count"] == 11


def test_exactly_seven_datasets():
    assert tuple(_load()["datasets"]["expected"]) == DATASETS
    assert _load()["datasets"]["expected_count"] == 7


def test_two_tracks_defined():
    t = _load()["tracks"]
    assert "controlled_architectural_track" in t
    assert "native_fidelity_reference_track" in t


def test_generation_disabled():
    assert _load()["scope"]["generation_enabled"] is False
    assert _load()["scope"]["retrieval_only"] is True


def test_tracks_not_merged_in_reporting():
    r = _load()["reporting_rules"]
    assert r["mixing_tracks_in_single_leaderboard"] is False
    assert r["mandatory_track_label"] is True