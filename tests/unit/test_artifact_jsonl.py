"""Tests for atomic JSONL artifact writing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sah_rag.artifacts.jsonl import (
    AtomicJsonlWriter,
    write_jsonl_atomic,
)


def test_atomic_writer_creates_jsonl(
    tmp_path: Path,
) -> None:
    """Successful writes should produce complete JSONL."""
    output = tmp_path / "records.jsonl"
    count = write_jsonl_atomic(
        output,
        [
            {"id": "a"},
            {"id": "b"},
        ],
    )
    assert count == 2
    assert output.is_file()
    lines = output.read_text(
        encoding="utf-8"
    ).splitlines()
    assert [
        json.loads(line)
        for line in lines
    ] == [
        {"id": "a"},
        {"id": "b"},
    ]


def test_atomic_writer_rejects_existing_output(
    tmp_path: Path,
) -> None:
    """Existing outputs require explicit overwrite."""
    output = tmp_path / "records.jsonl"
    output.write_text(
        '{"existing":true}\n',
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError), AtomicJsonlWriter(output):
        pass


def test_atomic_writer_removes_partial_output(
    tmp_path: Path,
) -> None:
    """Failed writes must not leave partial final artifacts."""
    output = tmp_path / "records.jsonl"
    temporary = tmp_path / ".records.jsonl.tmp"
    with pytest.raises(RuntimeError), AtomicJsonlWriter(output) as writer:
        writer.write({"id": "a"})
        raise RuntimeError("simulated failure")
    assert not output.exists()
    assert not temporary.exists()


def test_atomic_writer_can_overwrite(
    tmp_path: Path,
) -> None:
    """Explicit overwrite should replace existing contents."""
    output = tmp_path / "records.jsonl"
    output.write_text(
        '{"old":true}\n',
        encoding="utf-8",
    )
    write_jsonl_atomic(
        output,
        [{"new": True}],
        overwrite=True,
    )
    assert json.loads(
        output.read_text(encoding="utf-8")
    ) == {"new": True}