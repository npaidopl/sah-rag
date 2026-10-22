"""Tests for the Step 94 readiness sequence planner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNER = REPO_ROOT / "scripts" / "plan_architecture_readiness_sequence.py"
MANIFEST = REPO_ROOT / "baseline_audit" / "architecture_readiness_gap_manifest.yaml"
REGISTRY = REPO_ROOT / "baseline_audit" / "architecture_readiness_gap_registry.yaml"

ARCHS = ("sah_rag","longrag","raptor","hipporag2","lightrag","pathrag",
         "disretrieval","memorag","kag","metadata_rag","hikey")


def _run_planner() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PLANNER), "--no-update-manifest"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_planner_runs():
    r = _run_planner()
    assert r.returncode == 0, r.stderr


def test_rag_system_order_is_zero():
    a = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["architectures"]
    assert a["sah_rag"]["recommended_order"] == 0


def test_sequence_includes_all_architectures():
    a = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["architectures"]
    orders = sorted(rec["recommended_order"] for rec in a.values())
    assert orders == list(range(len(ARCHS)))


def test_priority_deterministic():
    r1 = _run_planner()
    r2 = _run_planner()
    # Extract the sequence section
    def _seq(text: str) -> list[str]:
        lines = []
        for line in text.splitlines():
            s = line.strip()
            if s and s[0].isdigit() and "." in s[:4]:
                parts = s.split(None, 1)
                if len(parts) == 2 and parts[1]:
                    name = parts[1].split()[0].rstrip("-").strip()
                    lines.append(name)
        return lines
    assert _seq(r1.stdout) == _seq(r2.stdout)


def test_first_target_is_controlled_track_eligible():
    m = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    t = m["next_gate"]["target_architecture"]
    a = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["architectures"]
    assert t is not None
    assert a[t]["controlled_track_status"] == "ELIGIBLE"


def test_planner_executes_no_architecture_code():
    src = PLANNER.read_text(encoding="utf-8")
    for token in ("importlib","__import__","exec(","eval(","subprocess.Popen",
                  "BGEM3FlagModel","QdrantClient"):
        assert token not in src, token