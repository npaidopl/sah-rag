"""Tests for exported embedding JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path

EMBEDDING_SCHEMA = Path("schemas/chunk_embedding.schema.json")

SHARD_HEADER_SCHEMA = Path("schemas/embedding_shard_header.schema.json")


def test_embedding_schemas_exist() -> None:
    """Both embedding schemas must be exported."""

    assert EMBEDDING_SCHEMA.is_file()
    assert EMBEDDING_SCHEMA.stat().st_size > 0

    assert SHARD_HEADER_SCHEMA.is_file()
    assert SHARD_HEADER_SCHEMA.stat().st_size > 0


def test_embedding_record_schema_requires_chunk_id() -> None:
    """Embedding records must retain chunk identity."""

    schema = json.loads(EMBEDDING_SCHEMA.read_text(encoding="utf-8"))

    assert "chunk_id" in schema["required"]


def test_shard_header_requires_configuration_hash() -> None:
    """Shard headers must identify their configuration."""

    schema = json.loads(SHARD_HEADER_SCHEMA.read_text(encoding="utf-8"))

    assert "embedding_config_hash" in schema["required"]
