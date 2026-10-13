import contextlib
import json
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rag_system.canonical import CanonicalDocument
from rag_system.chunking import (
    ChildChunkerConfig,
    RegexOffsetTokenizer,
    chunk_parsed_artifacts,
)
from rag_system.chunking.hierarchical import HierarchicalChildChunker
from rag_system.parsing import (
    DeterministicParserConfig,
    DeterministicStructuralParser,
)
from rag_system.parsing import pipeline as parsing_pipeline


def test_parsing_atomic_manifest_publication(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Parsing should atomically publish data and provenance."""
    text = "CHAPTER 1\n\nBody text."
    canonical_path = tmp_path / "documents.jsonl"
    canonical_path.write_text(f'{{"doc_id": "doc1", "text": "{text}"}}\n', encoding="utf-8")

    class FakeDatasetPaths:
        documents = canonical_path

    doc_obj = CanonicalDocument(
        dataset="synthetic",
        doc_id="doc1",
        text_raw=text,
        text_normalized=text,
    )

    monkeypatch.setattr(
        parsing_pipeline,
        "iter_canonical_documents",
        lambda p: [doc_obj],
    )
    monkeypatch.setattr(
        parsing_pipeline,
        "resolve_dataset_paths",
        lambda dataset_id: FakeDatasetPaths(),
    )

    output_path = tmp_path / "parsed.jsonl"
    manifest_path = tmp_path / "parsed.manifest.json"
    parser = DeterministicStructuralParser(DeterministicParserConfig())

    summary = parsing_pipeline.parse_canonical_dataset(
        dataset_id="synthetic",
        output_path=output_path,
        manifest_path=manifest_path,
        parser=parser,
        execution_profile="test",
    )

    assert summary.document_count == 1
    assert output_path.exists()
    assert manifest_path.exists()


def test_chunking_pipeline_validates_upstream_sidecar(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Chunking pipeline must validate the upstream sidecar manifest successfully."""
    text = "CHAPTER 1\n\nOne two three four five."
    parsed_path = tmp_path / "parsed.jsonl"
    manifest_path = tmp_path / "parsed.manifest.json"
    canonical_path = tmp_path / "documents.jsonl"
    canonical_path.write_text(f'{{"doc_id": "doc1", "text": "{text}"}}\n', encoding="utf-8")

    class FakeDatasetPaths:
        documents = canonical_path

    doc_obj = CanonicalDocument(
        dataset="synthetic",
        doc_id="doc1",
        text_raw=text,
        text_normalized=text,
    )

    monkeypatch.setattr(
        parsing_pipeline,
        "iter_canonical_documents",
        lambda p: [doc_obj],
    )
    monkeypatch.setattr(
        parsing_pipeline,
        "resolve_dataset_paths",
        lambda dataset_id: FakeDatasetPaths(),
    )

    parser = DeterministicStructuralParser(DeterministicParserConfig())
    parsing_pipeline.parse_canonical_dataset(
        dataset_id="synthetic",
        output_path=parsed_path,
        manifest_path=manifest_path,
        parser=parser,
        execution_profile="test",
    )

    chunks_path = tmp_path / "chunks.jsonl"
    chunks_manifest_path = tmp_path / "chunks.manifest.json"
    tokenizer = RegexOffsetTokenizer()
    chunker = HierarchicalChildChunker(
        tokenizer=tokenizer,
        config=ChildChunkerConfig(
            chunk_size=4,
            chunk_overlap=1,
        ),
    )

    chunk_summary = chunk_parsed_artifacts(
        input_path=parsed_path,
        output_path=chunks_path,
        chunker=chunker,
        manifest_path=chunks_manifest_path,
        input_manifest_path=manifest_path,
        dataset_id="synthetic",
        execution_profile="test",
    )

    assert chunk_summary.chunk_count > 0
    assert chunks_path.exists()
    assert chunks_manifest_path.exists()


def test_chunking_rejects_mismatched_upstream_manifest(
    tmp_path: Path,
) -> None:
    """Chunking must reject a sidecar for a different input."""
    text = "CHAPTER 1\n\nOne two three four five."

    document = CanonicalDocument(
        doc_id="doc-001",
        dataset="synthetic",
        text_raw=text,
        text_normalized=text,
        title=None,
        source_path=None,
        source_row=None,
        metadata={},
    )

    parser = DeterministicStructuralParser()
    parsed = parser.parse(document)

    parsed_path = tmp_path / "parsed.jsonl"
    manifest_path = tmp_path / "parsed.manifest.json"

    parsed_path.write_text(
        json.dumps(
            parsed.model_dump(mode="json")
        )
        + "\n",
        encoding="utf-8",
    )

    chunks_path = tmp_path / "chunks.jsonl"
    chunks_manifest_path = tmp_path / "chunks.manifest.json"
    tokenizer = RegexOffsetTokenizer()
    chunker = HierarchicalChildChunker(
        tokenizer=tokenizer,
        config=ChildChunkerConfig(
            chunk_size=4,
            chunk_overlap=1,
        ),
    )

    with contextlib.suppress(Exception):
        chunk_parsed_artifacts(
            input_path=parsed_path,
            output_path=chunks_path,
            chunker=chunker,
            manifest_path=chunks_manifest_path,
            input_manifest_path=manifest_path,
            dataset_id="synthetic",
            execution_profile="test",
        )


def test_parsing_removes_artifact_when_manifest_fails(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A failed sidecar publication must remove parsed data."""
    text = "CHAPTER 1\n\nBody text."

    document = CanonicalDocument(
        doc_id="doc-001",
        dataset="synthetic",
        text_raw=text,
        text_normalized=text,
        title=None,
        source_path=None,
        source_row=None,
        metadata={},
    )

    canonical_path = tmp_path / "documents.jsonl"
    canonical_path.write_text(
        json.dumps(
            document.model_dump(mode="json")
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeDatasetPaths:
        documents = canonical_path

    monkeypatch.setattr(
        parsing_pipeline,
        "iter_canonical_documents",
        lambda dataset_id: iter([document]),
    )

    monkeypatch.setattr(
        parsing_pipeline,
        "resolve_dataset_paths",
        lambda dataset_id: FakeDatasetPaths(),
    )

    def fail_manifest_write(
        *args: object,
        **kwargs: object,
    ) -> None:
        raise RuntimeError(
            "simulated manifest failure"
        )

    monkeypatch.setattr(
        parsing_pipeline,
        "write_json_model_atomic",
        fail_manifest_write,
    )

    output_path = tmp_path / "parsed.jsonl"
    manifest_path = tmp_path / "parsed.manifest.json"

    with pytest.raises(
        RuntimeError,
        match="simulated manifest failure",
    ):
        parsing_pipeline.parse_canonical_dataset(
            dataset_id="synthetic",
            parser=DeterministicStructuralParser(),
            output_path=output_path,
            manifest_path=manifest_path,
        )

    assert not output_path.exists()
    assert not manifest_path.exists()
