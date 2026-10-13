"""Integration tests for parsed-to-chunk streaming."""

from __future__ import annotations

from pathlib import Path

from rag_system.artifacts import ChildChunk
from rag_system.artifacts.jsonl import write_jsonl_atomic
from rag_system.canonical.jsonl import iter_jsonl
from rag_system.canonical.models import CanonicalDocument
from rag_system.chunking import (
    ChildChunkerConfig,
    HierarchicalChildChunker,
    RegexOffsetTokenizer,
    chunk_parsed_artifacts,
)
from rag_system.parsing import DeterministicStructuralParser


def test_chunk_parsed_artifacts_pipeline(
    tmp_path: Path,
) -> None:
    """Parsed documents should become validated child JSONL."""

    text = (
        "CHAPTER 1\n\n"
        "One two three four five six seven eight."
    )

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

    parsed = DeterministicStructuralParser().parse(
        document
    )

    parsed_path = tmp_path / "parsed.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"

    write_jsonl_atomic(
        parsed_path,
        [parsed],
    )

    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(
            chunk_size=5,
            chunk_overlap=1,
        ),
    )

    summary = chunk_parsed_artifacts(
        input_path=parsed_path,
        output_path=chunks_path,
        chunker=chunker,
    )

    chunks = [
        ChildChunk.model_validate(record)
        for record in iter_jsonl(chunks_path)
    ]

    assert summary.parsed_document_count == 1
    assert summary.parent_count == 1
    assert summary.chunk_count == len(chunks)
    assert summary.parents_without_chunks == 0
    assert chunks
    assert all(
        chunk.parent_id
        == parsed.parents[0].parent_id
        for chunk in chunks
    )
