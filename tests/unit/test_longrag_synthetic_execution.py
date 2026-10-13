"""Tests for the LongRAG synthetic execution outputs (long units + embeddings)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

OUTPUT_ROOT = Path(
    "outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/longrag"
)
CANONICAL = Path("processed/canonical/infrastructure-pilot-synthetic-v1")


@pytest.fixture(scope="module")
def long_units() -> list[dict]:
    p = OUTPUT_ROOT / "units" / "long_units.jsonl"
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def embedding_records() -> list[dict]:
    p = OUTPUT_ROOT / "embeddings" / "embedding_records.jsonl"
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def documents() -> list[dict]:
    p = CANONICAL / "documents.jsonl"
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_twelve_documents_loaded(documents: list[dict]) -> None:
    assert len(documents) == 12


def test_long_units_are_nonempty(long_units: list[dict]) -> None:
    assert len(long_units) > 0
    for lu in long_units:
        assert lu["text"].strip()
        assert lu["token_count"] > 0


def test_every_document_has_lineage(
    long_units: list[dict], documents: list[dict]
) -> None:
    covered: set[str] = set()
    for lu in long_units:
        covered.update(lu["member_doc_ids"])
    expected = {d["doc_id"] for d in documents}
    assert expected.issubset(covered)


def test_long_unit_ids_are_unique(long_units: list[dict]) -> None:
    ids = [lu["long_unit_id"] for lu in long_units]
    assert len(ids) == len(set(ids))


def test_embedding_records_have_one_per_long_unit(
    long_units: list[dict], embedding_records: list[dict]
) -> None:
    assert len(embedding_records) == len(long_units)
    lu_ids = {lu["long_unit_id"] for lu in long_units}
    em_ids = {r["long_unit_id"] for r in embedding_records}
    assert lu_ids == em_ids