"""Tests for typed canonical streaming loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sah_rag.canonical.loaders import (
    CanonicalModelValidationError,
    iter_validated_models,
)
from sah_rag.canonical.models import CanonicalDocument


def write_jsonl(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    """Write records as JSONL."""

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False)
            )
            handle.write("\n")


def test_iter_validated_models_streams_documents(
    tmp_path: Path,
) -> None:
    """Typed documents should be yielded in input order."""

    path = tmp_path / "documents.jsonl"

    write_jsonl(
        path,
        [
            {
                "doc_id": "doc-1",
                "dataset": "test",
                "text_raw": "First.",
                "text_normalized": "First.",
            },
            {
                "doc_id": "doc-2",
                "dataset": "test",
                "text_raw": "Second.",
                "text_normalized": "Second.",
            },
        ],
    )

    documents = list(
        iter_validated_models(
            path,
            CanonicalDocument,
        )
    )

    assert [
        document.doc_id
        for document in documents
    ] == ["doc-1", "doc-2"]


def test_iter_validated_models_reports_record_number(
    tmp_path: Path,
) -> None:
    """Validation errors should identify the record number."""

    path = tmp_path / "documents.jsonl"

    write_jsonl(
        path,
        [
            {
                "doc_id": "doc-1",
                "dataset": "test",
                "text_raw": "First.",
                "text_normalized": "First.",
            },
            {
                "doc_id": "",
                "dataset": "test",
                "text_raw": "Second.",
                "text_normalized": "Second.",
            },
        ],
    )

    with pytest.raises(
        CanonicalModelValidationError
    ) as exc_info:
        list(
            iter_validated_models(
                path,
                CanonicalDocument,
            )
        )

    message = str(exc_info.value)

    assert "record 2" in message
    assert "CanonicalDocument" in message