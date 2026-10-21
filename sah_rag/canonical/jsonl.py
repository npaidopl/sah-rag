"""Streaming JSONL utilities."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


class JsonlReadError(ValueError):
    """Raised when a JSONL record cannot be decoded."""

    def __init__(
        self,
        path: Path,
        line_number: int,
        message: str,
    ) -> None:
        self.path = path
        self.line_number = line_number

        super().__init__(
            f"{path}: invalid JSON on line {line_number}: {message}"
        )


def iter_jsonl(path: Path) -> Iterator[JsonObject]:
    """Yield JSON objects from a JSONL file without loading it fully."""

    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()

            if not stripped:
                continue

            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise JsonlReadError(
                    path=path,
                    line_number=line_number,
                    message=str(exc),
                ) from exc

            if not isinstance(value, dict):
                raise JsonlReadError(
                    path=path,
                    line_number=line_number,
                    message=(
                        "Each JSONL record must be a JSON object, "
                        f"received {type(value).__name__}."
                    ),
                )

            yield value


def read_first_record(path: Path) -> JsonObject | None:
    """Return the first non-empty JSONL object or None."""

    return next(iter_jsonl(path), None)


def count_jsonl_records(path: Path) -> int:
    """Count valid non-empty JSONL records through streaming iteration."""

    return sum(1 for _ in iter_jsonl(path))