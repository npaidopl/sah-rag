"""Tests for pilot retrieval persistence (Step 87)."""

from __future__ import annotations

import json
from pathlib import Path

from rag_system.retrieval.document_resolution import RetrievedDocument
from rag_system.retrieval.pilot_persistence import (
    read_pilot_retrieval_artifacts,
    write_pilot_retrieval_artifacts,
)


def _doc(doc_id: str, rank: int, query_id: str = "q-1") -> RetrievedDocument:
    return RetrievedDocument(
        query_id=query_id,
        doc_id=doc_id,
        dataset="infrastructure-pilot-synthetic-v1",
        rank=rank,
        score=1.0 - 0.1 * rank,
        best_parent_rank=rank,
        contributing_parent_ids=(f"p-{rank}",),
        contributing_chunk_ids=(f"c-{rank}",),
        metadata={},
    )


def _write(tmp_path: Path) -> Path:
    root = tmp_path / "retrieval"
    write_pilot_retrieval_artifacts(
        root=root,
        query_embeddings=[
            {
                "query_id": f"q-{i}",
                "query_text_sha256": "a" * 64,
                "model_id": "BAAI/bge-m3",
                "model_revision": "x" * 40,
                "dense_vector": [0.0] * 1024,
                "sparse_indices": [0, 1],
                "sparse_values": [0.5, 0.3],
                "dense_normalized": True,
                "metadata": {},
            }
            for i in range(1, 7)
        ],
        ranked_children=[
            {
                "query_id": "q-1",
                "retrieval_method": "hybrid_rrf",
                "chunk_id": "c-1",
                "parent_id": "p-1",
                "doc_id": "d-1",
                "rank": 1,
                "score": 0.9,
                "metadata": {},
            }
        ],
        ranked_parents=[
            {
                "query_id": "q-1",
                "parent_id": "p-1",
                "doc_id": "d-1",
                "dataset": "infrastructure-pilot-synthetic-v1",
                "rank": 1,
                "score": 0.9,
                "best_chunk_rank": 1,
                "matched_chunk_count": 1,
                "contributing_chunk_ids": ["c-1"],
                "child_scores": [0.9],
                "metadata": {},
            }
        ],
        ranked_documents=[_doc("d-1", 1)],
        manifest={"artifact_type": "pilot_retrieval_manifest", "version": "1.0"},
    )
    return root


def test_round_trip_succeeds(tmp_path: Path):
    root = _write(tmp_path)
    loaded = read_pilot_retrieval_artifacts(root=root, expected_query_count=6)
    assert len(loaded["query_embeddings"]) == 6
    assert loaded["checksums_valid"] is True


def test_checksums_validate(tmp_path: Path):
    root = _write(tmp_path)
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    assert checksums["algorithm"] == "sha256"
    assert "query_embeddings.jsonl" in checksums["files"]


def test_existing_output_rejected_without_overwrite(tmp_path: Path):
    root = _write(tmp_path)
    try:
        write_pilot_retrieval_artifacts(
            root=root,
            query_embeddings=[],
            ranked_children=[],
            ranked_parents=[],
            ranked_documents=[],
            manifest={},
        )
    except FileExistsError:
        return
    raise AssertionError("expected FileExistsError")


def test_overwrite_allows_rewrite(tmp_path: Path):
    root = _write(tmp_path)
    write_pilot_retrieval_artifacts(
        root=root,
        query_embeddings=[
            {
                "query_id": f"q-{i}",
                "query_text_sha256": "b" * 64,
                "model_id": "BAAI/bge-m3",
                "model_revision": "y" * 40,
                "dense_vector": [0.0] * 1024,
                "sparse_indices": [],
                "sparse_values": [],
                "dense_normalized": True,
                "metadata": {},
            }
            for i in range(1, 7)
        ],
        ranked_children=[],
        ranked_parents=[],
        ranked_documents=[],
        manifest={},
        overwrite=True,
    )
    loaded = read_pilot_retrieval_artifacts(root=root, expected_query_count=6)
    assert len(loaded["query_embeddings"]) == 6


def test_query_ids_must_be_six(tmp_path: Path):
    root = _write(tmp_path)
    try:
        read_pilot_retrieval_artifacts(root=root, expected_query_count=99)
    except ValueError:
        return
    raise AssertionError("expected ValueError")