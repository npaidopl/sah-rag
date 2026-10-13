"""Tests for streaming artifact fingerprints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rag_system.artifacts import (
    count_nonempty_lines,
    file_size_bytes,
    sha256_file,
)


def test_sha256_file_matches_hashlib(
    tmp_path: Path,
) -> None:
    """Streaming SHA-256 should match direct hashing."""

    path = tmp_path / "artifact.bin"
    content = b"artifact-content\n"

    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(
        content
    ).hexdigest()


def test_file_size_bytes(
    tmp_path: Path,
) -> None:
    """Exact file size should be returned."""

    path = tmp_path / "artifact.txt"
    path.write_bytes(b"12345")

    assert file_size_bytes(path) == 5


def test_count_nonempty_lines(
    tmp_path: Path,
) -> None:
    """Empty lines should not be counted as records."""

    path = tmp_path / "records.jsonl"
    path.write_bytes(
        b'{"id":1}\n\n{"id":2}\n'
    )

    assert count_nonempty_lines(path) == 2


def test_missing_file_raises(
    tmp_path: Path,
) -> None:
    """Missing fingerprints should fail explicitly."""

    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing.jsonl")


def test_invalid_buffer_size(
    tmp_path: Path,
) -> None:
    """Buffer sizes must remain positive."""

    path = tmp_path / "artifact.txt"
    path.write_text(
        "content",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        sha256_file(
            path,
            buffer_size=0,
        )
