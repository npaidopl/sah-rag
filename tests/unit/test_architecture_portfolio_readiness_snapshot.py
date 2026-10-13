"""Tests for the architecture portfolio readiness snapshot validator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VALIDATOR = (
    REPO_ROOT
    / "scripts"
    / "validate_architecture_portfolio_readiness_snapshot.py"
)


def test_validator_passes_on_current_snapshot() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "[OK] portfolio readiness snapshot" in result.stdout