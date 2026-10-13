"""Tests for the infrastructure pilot planner (preview only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNER = REPO_ROOT / "scripts" / "plan_infrastructure_pilot.py"
PILOT_ROOT = REPO_ROOT / "outputs" / "infrastructure_pilot"


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PLANNER)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def _listing() -> list[str]:
    if not PILOT_ROOT.exists():
        return []
    return sorted(
        str(p.relative_to(PILOT_ROOT)).replace("\\", "/")
        for p in PILOT_ROOT.rglob("*")
    )


def test_planner_runs_cleanly():
    r = _run()
    assert r.returncode == 0, r.stderr
    assert "[OK] Infrastructure pilot plan preview" in r.stdout


def test_preview_ids_are_deterministic():
    a = _run().stdout
    b = _run().stdout
    assert a == b


def test_four_unique_preview_cell_ids():
    out = _run().stdout
    cells = [line for line in out.splitlines() if "pilot-preview-cell-" in line]
    assert len(cells) == 4
    assert len(set(cells)) == 4


def test_four_unique_preview_plan_ids():
    out = _run().stdout
    plans = [line for line in out.splitlines() if "pilot-preview-plan-" in line]
    assert len(plans) == 4
    assert len(set(plans)) == 4


def test_planner_does_not_create_or_modify_output_directory():
    """The preview planner is pure print-only.

    It must not create the pilot output root, and if the root already
    exists (from prior materialization steps), it must not modify it.
    This test never deletes the real project output tree.
    """
    existed_before = PILOT_ROOT.exists()
    listing_before = _listing()

    _run()

    existed_after = PILOT_ROOT.exists()
    listing_after = _listing()

    assert existed_after == existed_before, (
        "Planner must not change whether the pilot output root exists"
    )
    assert listing_after == listing_before, (
        "Planner must not modify the contents of the pilot output root"
    )