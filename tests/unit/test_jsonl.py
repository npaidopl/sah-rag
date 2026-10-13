"""Tests for streaming JSONL utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_system.canonical.jsonl import (
    JsonlReadError,
    count_jsonl_records,
    iter_jsonl,
    read_first_record,
)


def write_lines(path: Path, values: list[object]) -> None:
    """Write JSON values as JSONL."""

    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False)
            )
            handle.write("\n")


def test_iter_jsonl_streams_objects(tmp_path: Path) -> None:
    """JSONL records should be yielded in order."""

    path = tmp_path / "records.jsonl"

    write_lines(
        path,
        [
            {"id": "a"},
            {"id": "b"},
        ],
    )

    records = list(iter_jsonl(path))

    assert records == [
        {"id": "a"},
        {"id": "b"},
    ]


def test_read_first_record(tmp_path: Path) -> None:
    """The first non-empty record should be returned."""

    path = tmp_path / "records.jsonl"

    path.write_text(
        '\n{"id": "first"}\n{"id": "second"}\n',
        encoding="utf-8",
    )

    assert read_first_record(path) == {"id": "first"}


def test_count_jsonl_records(tmp_path: Path) -> None:
    """Record counting should ignore empty lines."""

    path = tmp_path / "records.jsonl"

    path.write_text(
        '{"id": 1}\n\n{"id": 2}\n',
        encoding="utf-8",
    )

    assert count_jsonl_records(path) == 2


def test_iter_jsonl_rejects_arrays(tmp_path: Path) -> None:
    """Each JSONL line must contain an object."""

    path = tmp_path / "records.jsonl"

    path.write_text(
        '["not", "an", "object"]\n',
        encoding="utf-8",
    )

    with pytest.raises(JsonlReadError):
        list(iter_jsonl(path))


def test_iter_jsonl_reports_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON should include file and line information."""

    path = tmp_path / "records.jsonl"

    path.write_text(
        '{"valid": true}\n{"invalid": }\n',
        encoding="utf-8",
    )

    with pytest.raises(JsonlReadError) as exc_info:
        list(iter_jsonl(path))

    message = str(exc_info.value)

    assert "line 2" in message
    assert str(path) in message