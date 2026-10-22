"""Tests for pilot canonical adapter persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from sah_rag.canonical.pilot_adapter import (
    ADAPTER_ID,
    adapt_pilot_dataset_to_canonical,
)
from sah_rag.canonical.pilot_adapter_persistence import (
    read_pilot_canonical_adapter,
    write_pilot_canonical_adapter,
)

SOURCE = Path("data/pilot/infrastructure-pilot-synthetic-v1")


def _fixture():
    return adapt_pilot_dataset_to_canonical(pilot_root=SOURCE)


def test_round_trip_succeeds(tmp_path: Path):
    documents, examples, validation, mapping = _fixture()
    write_pilot_canonical_adapter(
        target_root=tmp_path,
        documents=documents,
        examples=examples,
        validation=validation,
        mapping=mapping,
    )
    loaded = read_pilot_canonical_adapter(
        target_root=tmp_path, expected_adapter_id=ADAPTER_ID,
    )
    assert len(loaded["documents"]) == 12
    assert len(loaded["examples"]) == 6


def test_existing_target_rejected_without_overwrite(tmp_path: Path):
    documents, examples, validation, mapping = _fixture()
    write_pilot_canonical_adapter(
        target_root=tmp_path,
        documents=documents,
        examples=examples,
        validation=validation,
        mapping=mapping,
    )
    with pytest.raises(FileExistsError):
        write_pilot_canonical_adapter(
            target_root=tmp_path,
            documents=documents,
            examples=examples,
            validation=validation,
            mapping=mapping,
        )


def test_overwrite_succeeds_with_flag(tmp_path: Path):
    documents, examples, validation, mapping = _fixture()
    write_pilot_canonical_adapter(
        target_root=tmp_path,
        documents=documents,
        examples=examples,
        validation=validation,
        mapping=mapping,
    )
    write_pilot_canonical_adapter(
        target_root=tmp_path,
        documents=documents,
        examples=examples,
        validation=validation,
        mapping=mapping,
        overwrite=True,
    )


def test_checksum_mismatch_rejected(tmp_path: Path):
    documents, examples, validation, mapping = _fixture()
    write_pilot_canonical_adapter(
        target_root=tmp_path,
        documents=documents,
        examples=examples,
        validation=validation,
        mapping=mapping,
    )
    p = tmp_path / "documents.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        read_pilot_canonical_adapter(target_root=tmp_path)


def test_adapter_id_mismatch_rejected(tmp_path: Path):
    documents, examples, validation, mapping = _fixture()
    write_pilot_canonical_adapter(
        target_root=tmp_path,
        documents=documents,
        examples=examples,
        validation=validation,
        mapping=mapping,
    )
    with pytest.raises(RuntimeError, match="adapter_id mismatch"):
        read_pilot_canonical_adapter(
            target_root=tmp_path, expected_adapter_id="something-else",
        )


def test_deterministic_bytes_across_two_targets(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    documents, examples, validation, mapping = _fixture()
    for root in (a, b):
        write_pilot_canonical_adapter(
            target_root=root,
            documents=documents,
            examples=examples,
            validation=validation,
            mapping=mapping,
        )
    for name in (
        "documents.jsonl",
        "examples.jsonl",
        "validation.json",
        "manifest.json",
        "adapter.mapping.json",
        "checksums.json",
    ):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
