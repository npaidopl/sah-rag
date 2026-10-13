"""Tests for the exported artifact-sidecar JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_PATH = Path(
    "schemas/artifact_sidecar_manifest.schema.json"
)


def test_sidecar_schema_exists() -> None:
    """The sidecar schema must be exported."""

    assert SCHEMA_PATH.is_file()
    assert SCHEMA_PATH.stat().st_size > 0


def test_sidecar_schema_contains_required_identity() -> None:
    """Core artifact identity fields must be required."""

    schema = json.loads(
        SCHEMA_PATH.read_text(encoding="utf-8")
    )

    required = set(schema["required"])

    assert {
        "artifact_type",
        "artifact_path",
        "artifact_sha256",
        "artifact_size_bytes",
        "record_count",
        "component",
    }.issubset(required)
