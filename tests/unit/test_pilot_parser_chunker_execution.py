"""Execution invariants for the pilot parser+chunker."""

from __future__ import annotations

import os
from pathlib import Path

from sah_rag.artifacts import ChildChunk, ParsedDocument
from sah_rag.chunking import (
    ChildChunkerConfig,
    HierarchicalChildChunker,
    RegexOffsetTokenizer,
    chunk_parsed_artifacts,
)
from sah_rag.parsing import (
    DeterministicStructuralParser,
    parse_canonical_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _prepare(tmp_path: Path):
    canonical_root = (REPO_ROOT / "processed" / "canonical").resolve()
    os.environ["RAG_CANONICAL_ROOT"] = str(canonical_root)
    parsed_path = tmp_path / "parsed.jsonl"
    parse_canonical_dataset(
        dataset_id="infrastructure-pilot-synthetic-v1",
        parser=DeterministicStructuralParser(),
        output_path=parsed_path,
        overwrite=True,
    )
    return parsed_path


def test_parser_covers_all_documents(tmp_path: Path):
    parsed_path = _prepare(tmp_path)
    parsed = [
        ParsedDocument.model_validate_json(line)
        for line in parsed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(parsed) == 12


def test_chunker_produces_unique_ids(tmp_path: Path):
    parsed_path = _prepare(tmp_path)
    chunks_path = tmp_path / "chunks.jsonl"
    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(chunk_size=256, chunk_overlap=32),
    )
    chunk_parsed_artifacts(
        input_path=parsed_path,
        output_path=chunks_path,
        chunker=chunker,
        overwrite=True,
    )
    chunks = [
        ChildChunk.model_validate_json(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunks_cover_all_documents(tmp_path: Path):
    parsed_path = _prepare(tmp_path)
    chunks_path = tmp_path / "chunks.jsonl"
    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(chunk_size=256, chunk_overlap=32),
    )
    chunk_parsed_artifacts(
        input_path=parsed_path, output_path=chunks_path,
        chunker=chunker, overwrite=True,
    )
    chunk_doc_ids = {
        ChildChunk.model_validate_json(line).doc_id
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert len(chunk_doc_ids) == 12


def test_parent_ids_reference_real_parents(tmp_path: Path):
    parsed_path = _prepare(tmp_path)
    parsed = [
        ParsedDocument.model_validate_json(line)
        for line in parsed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parent_ids = {p.parent_id for d in parsed for p in d.parents}
    chunks_path = tmp_path / "chunks.jsonl"
    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(chunk_size=256, chunk_overlap=32),
    )
    chunk_parsed_artifacts(
        input_path=parsed_path, output_path=chunks_path,
        chunker=chunker, overwrite=True,
    )
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = ChildChunk.model_validate_json(line)
        assert c.parent_id in parent_ids


def test_duplicate_documents_stay_distinct(tmp_path: Path):
    parsed_path = _prepare(tmp_path)
    parsed = [
        ParsedDocument.model_validate_json(line)
        for line in parsed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    doc_ids = [d.doc_id for d in parsed]
    assert len(doc_ids) == 12
    assert len(set(doc_ids)) == 12


def test_no_embedding_files_created(tmp_path: Path):
    parsed_path = _prepare(tmp_path)
    chunks_path = tmp_path / "chunks.jsonl"
    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(chunk_size=256, chunk_overlap=32),
    )
    chunk_parsed_artifacts(
        input_path=parsed_path, output_path=chunks_path,
        chunker=chunker, overwrite=True,
    )
    banned = ("embeddings.jsonl", "embedding_records.jsonl",
              "retrieval_cases.jsonl", "ranked_items.jsonl",
              "query_metrics.jsonl", "aggregate_metrics.json")
    for p in tmp_path.rglob("*"):
        assert p.name not in banned


def test_source_order_non_negative(tmp_path: Path):
    parsed_path = _prepare(tmp_path)
    chunks_path = tmp_path / "chunks.jsonl"
    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(chunk_size=256, chunk_overlap=32),
    )
    chunk_parsed_artifacts(
        input_path=parsed_path, output_path=chunks_path,
        chunker=chunker, overwrite=True,
    )
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = ChildChunk.model_validate_json(line)
        assert c.chunk_index >= 0
        assert c.parent_chunk_index >= 0
