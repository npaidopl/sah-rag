"""Streaming JSONL writers for derived artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

JsonObject = dict[str, Any]


def serialize_record(
    record: BaseModel | JsonObject,
) -> JsonObject:
    """Convert one supported record to a JSON-compatible object."""
    if isinstance(record, BaseModel):
        return record.model_dump(
            mode="json",
            exclude_none=False,
        )
    if isinstance(record, dict):
        return record
    raise TypeError(
        "JSONL records must be Pydantic models or dictionaries."
    )


class AtomicJsonlWriter:
    """Write JSONL through a temporary file and atomic replacement."""

    def __init__(
        self,
        path: Path,
        *,
        overwrite: bool = False,
    ) -> None:
        self.path = path
        self.overwrite = overwrite
        self.temporary_path = path.with_name(
            f".{path.name}.tmp"
        )
        self._handle: Any | None = None
        self._record_count = 0
        self._committed = False

    @property
    def record_count(self) -> int:
        """Return the number of written records."""
        return self._record_count

    def __enter__(self) -> AtomicJsonlWriter:
        """Open the temporary output file."""
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Output file already exists: {self.path}"
            )
        if self.temporary_path.exists():
            self.temporary_path.unlink()
        self._handle = self.temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        )
        return self

    def write(
        self,
        record: BaseModel | JsonObject,
    ) -> None:
        """Write one JSON object as one JSONL record."""
        if self._handle is None:
            raise RuntimeError(
                "AtomicJsonlWriter is not open."
            )
        serialized = json.dumps(
            serialize_record(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._handle.write(serialized)
        self._handle.write("\n")
        self._record_count += 1

    def commit(self) -> None:
        """Flush and atomically expose the completed output."""
        if self._handle is None:
            raise RuntimeError(
                "AtomicJsonlWriter is not open."
            )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None
        os.replace(
            self.temporary_path,
            self.path,
        )
        self._committed = True

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> None:
        """Commit on success and remove incomplete files on failure."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if exception_type is None and not self._committed:
            os.replace(
                self.temporary_path,
                self.path,
            )
            self._committed = True
            return
        if self.temporary_path.exists():
            self.temporary_path.unlink()


def write_jsonl_atomic(
    path: Path,
    records: Iterable[BaseModel | JsonObject],
    *,
    overwrite: bool = False,
) -> int:
    """Write an iterable to JSONL and return its record count."""
    with AtomicJsonlWriter(
        path,
        overwrite=overwrite,
    ) as writer:
        for record in records:
            writer.write(record)
    return writer.record_count