"""Integration tests for streaming parsed artifacts."""

from __future__ import annotations

from pathlib import Path

from sah_rag.artifacts.models import ParsedDocument
from sah_rag.canonical.jsonl import iter_jsonl
from sah_rag.parsing import (
    DeterministicStructuralParser,
    pipeline,
)


def test_parse_canonical_dataset_streams_artifacts(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    """The parsing pipeline should write validated artifacts."""

    from _pytest.monkeypatch import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    records = [
        {
            "doc_id": "doc-001",
            "dataset": "synthetic",
            "text_raw": "CHAPTER 1\n\nBody text.",
            "text_normalized": "CHAPTER 1\n\nBody text.",
            "title": None,
            "source_path": None,
            "source_row": 0,
            "metadata": {},
        },
        {
            "doc_id": "doc-002",
            "dataset": "synthetic",
            "text_raw": "First.\n\nSecond.",
            "text_normalized": "First.\n\nSecond.",
            "title": None,
            "source_path": None,
            "source_row": 1,
            "metadata": {},
        },
    ]

    def fake_document_loader(
        dataset_id: str,
    ) -> object:
        assert dataset_id == "synthetic"

        from sah_rag.canonical.models import (
            CanonicalDocument,
        )

        return (
            CanonicalDocument.model_validate(record)
            for record in records
        )

    monkeypatch.setattr(
        pipeline,
        "iter_canonical_documents",
        fake_document_loader,
    )

    output = tmp_path / "parsed.jsonl"

    summary = pipeline.parse_canonical_dataset(
        dataset_id="synthetic",
        parser=DeterministicStructuralParser(),
        output_path=output,
    )

    assert summary.document_count == 2
    assert summary.parent_count >= 2
    assert summary.empty_document_count == 0

    parsed_records = [
        ParsedDocument.model_validate(record)
        for record in iter_jsonl(output)
    ]

    assert len(parsed_records) == 2
    assert parsed_records[0].doc_id == "doc-001"
    assert parsed_records[1].doc_id == "doc-002"