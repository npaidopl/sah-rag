"""Streaming fingerprints for files and artifact records."""

from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_BUFFER_SIZE = 1024 * 1024


def sha256_file(
    path: Path,
    *,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> str:
    """Calculate the SHA-256 fingerprint of a file."""

    if buffer_size < 1:
        raise ValueError(
            "buffer_size must be positive."
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(buffer_size):
            digest.update(chunk)

    return digest.hexdigest()


def file_size_bytes(path: Path) -> int:
    """Return the exact size of an existing file."""

    if not path.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    return path.stat().st_size


def count_nonempty_lines(path: Path) -> int:
    """Count non-empty binary lines without loading the file."""

    if not path.is_file():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    count = 0

    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1

    return count
