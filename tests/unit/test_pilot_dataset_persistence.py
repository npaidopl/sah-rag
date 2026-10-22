"""Tests for pilot dataset persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from datasets.pilot_synthetic import (
    DATASET_ID,
    DATASET_REVISION,
    build_infrastructure_pilot_dataset,
)
from datasets.pilot_persistence import (
    read_infrastructure_pilot_dataset,
    write_infrastructure_pilot_dataset,
)


def _fixture():
    docs, chunks, queries, judgments = build_infrastructure_pilot_dataset()
    return docs, chunks, queries, judgments


def test_round_trip_succeeds(tmp_path: Path):
    docs, chunks, queries, judgments = _fixture()
    write_infrastructure_pilot_dataset(
        root=tmp_path, documents=docs, chunks=chunks,
        queries=queries, judgments=judgments,
    )
    loaded = read_infrastructure_pilot_dataset(
        root=tmp_path,
        expected_dataset_id=DATASET_ID,
        expected_dataset_revision=DATASET_REVISION,
    )
    assert len(loaded["documents"]) == 12
    assert len(loaded["chunks"]) == 24
    assert len(loaded["queries"]) == 6
    assert len(loaded["judgments"]) == 12


def test_existing_dataset_rejected_without_overwrite(tmp_path: Path):
    docs, chunks, queries, judgments = _fixture()
    write_infrastructure_pilot_dataset(
        root=tmp_path, documents=docs, chunks=chunks,
        queries=queries, judgments=judgments,
    )
    with pytest.raises(FileExistsError):
        write_infrastructure_pilot_dataset(
            root=tmp_path, documents=docs, chunks=chunks,
            queries=queries, judgments=judgments,
        )


def test_overwrite_succeeds_with_flag(tmp_path: Path):
    docs, chunks, queries, judgments = _fixture()
    write_infrastructure_pilot_dataset(
        root=tmp_path, documents=docs, chunks=chunks,
        queries=queries, judgments=judgments,
    )
    write_infrastructure_pilot_dataset(
        root=tmp_path, documents=docs, chunks=chunks,
        queries=queries, judgments=judgments, overwrite=True,
    )


def test_checksum_mismatch_rejected(tmp_path: Path):
    docs, chunks, queries, judgments = _fixture()
    write_infrastructure_pilot_dataset(
        root=tmp_path, documents=docs, chunks=chunks,
        queries=queries, judgments=judgments,
    )
    p = tmp_path / "documents.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        read_infrastructure_pilot_dataset(root=tmp_path)


def test_dataset_id_mismatch_rejected(tmp_path: Path):
    docs, chunks, queries, judgments = _fixture()
    write_infrastructure_pilot_dataset(
        root=tmp_path, documents=docs, chunks=chunks,
        queries=queries, judgments=judgments,
    )
    with pytest.raises(RuntimeError, match="dataset_id mismatch"):
        read_infrastructure_pilot_dataset(
            root=tmp_path, expected_dataset_id="something-else",
        )


def test_revision_mismatch_rejected(tmp_path: Path):
    docs, chunks, queries, judgments = _fixture()
    write_infrastructure_pilot_dataset(
        root=tmp_path, documents=docs, chunks=chunks,
        queries=queries, judgments=judgments,
    )
    with pytest.raises(RuntimeError, match="dataset_revision mismatch"):
        read_infrastructure_pilot_dataset(
            root=tmp_path, expected_dataset_revision="9.9.9",
        )


def test_deterministic_bytes_across_two_roots(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    docs, chunks, queries, judgments = _fixture()
    for root in (a, b):
        write_infrastructure_pilot_dataset(
            root=root, documents=docs, chunks=chunks,
            queries=queries, judgments=judgments,
        )
    for name in (
        "documents.jsonl", "chunks.jsonl",
        "queries.jsonl", "judgments.jsonl",
        "dataset.manifest.json", "checksums.json",
    ):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_materialization_does_not_create_outputs(tmp_path: Path):
    docs, chunks, queries, judgments = _fixture()
    write_infrastructure_pilot_dataset(
        root=tmp_path, documents=docs, chunks=chunks,
        queries=queries, judgments=judgments,
    )
    assert not (tmp_path / "outputs").exists()
