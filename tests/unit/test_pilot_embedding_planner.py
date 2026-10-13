"""Tests for the pilot embedding planner output."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNER = REPO_ROOT / "scripts" / "plan_pilot_embedding.py"


def _run() -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PLANNER)],
                          cwd=REPO_ROOT, capture_output=True, text=True)


def test_planner_runs_cleanly():
    r = _run()
    assert r.returncode == 0, r.stderr
    assert "[OK] Pilot embedding plan" in r.stdout


def test_planner_reports_embedding_not_approved():
    out = _run().stdout
    assert "embedding execution approved: False" in out
