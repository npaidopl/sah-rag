"""Tests for LongRAG synthetic persistence, ground-truth guard, and
evaluation-case construction."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

OUTPUT_ROOT = Path(
    "outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/longrag"
)


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    p = OUTPUT_ROOT / "evaluation_cases" / "retrieval_cases.jsonl"
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def manifest() -> dict:
    p = OUTPUT_ROOT / "retrieval" / "retrieval.manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_six_query_embeddings(cases: list[dict]) -> None:
    assert len(cases) == 6


def test_retrieval_covers_every_query(cases: list[dict]) -> None:
    qids = [c["query_id"] for c in cases]
    assert len(qids) == len(set(qids))
    for c in cases:
        assert c["ranked_doc_ids"], f"empty ranked_doc_ids for {c['query_id']}"


def test_canonical_expansion_is_valid(cases: list[dict]) -> None:
    for c in cases:
        assert len(c["relevant_doc_ids"]) == 2
        assert len(set(c["relevant_doc_ids"])) == 2


def test_ground_truth_not_loaded_during_retrieval(manifest: dict) -> None:
    assert manifest["ground_truth_loaded_during_retrieval"] is False


def test_evaluation_case_count_is_six(cases: list[dict]) -> None:
    assert len(cases) == 6