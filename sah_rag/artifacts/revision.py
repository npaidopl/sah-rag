"""Source-code revision discovery."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_revision(
    repository_root: Path | None = None,
) -> str | None:
    """Return the current Git commit hash when available."""

    working_directory = (
        repository_root
        if repository_root is not None
        else Path.cwd()
    )

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    revision = result.stdout.strip()

    return revision or None
