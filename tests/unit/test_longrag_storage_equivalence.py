"""Tests for storage equivalence between the exact reference backend
and Qdrant local persistence on the frozen synthetic fixture."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

OUTPUT_ROOT = Path(
    "outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/longrag"
)
REPORT = Path("baseline_audit/longrag_storage_equivalence_report.yaml")


@pytest.fixture(scope="module")
def report() -> dict:
    import yaml
    return yaml.safe_load(REPORT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ref_rankings() -> list[dict]:
    path = OUTPUT_ROOT / "retrieval" / "reference_rankings.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def qdr_rankings() -> list[dict]:
    path = OUTPUT_ROOT / "retrieval" / "qdrant_rankings.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_report_passed(report: dict) -> None:
    assert report["comparison"]["status"] == "PASSED"
    assert report["conclusion"]["classification"] == (
        "SEMANTICALLY_NEUTRAL_FOR_CONTROLLED_TRACK"
    )
    assert report["conclusion"]["substitution_allowed"] is True


def test_point_count_matches(report: dict) -> None:
    r = report["results"]
    assert r["point_count_reference"] == r["point_count_qdrant"]


def test_top_1_overlap_is_exact(report: dict) -> None:
    assert report["results"]["top_1_overlap_mean"] == 1.0


def test_top_5_overlap_is_exact(report: dict) -> None:
    assert report["results"]["top_5_overlap_mean"] == 1.0


def test_top_10_overlap_is_exact(report: dict) -> None:
    assert report["results"]["top_10_overlap_mean"] == 1.0


def test_rank_order_is_identical(ref_rankings: list[dict], qdr_rankings: list[dict]) -> None:
    ref_by_q: dict[str, list[tuple[int, str]]] = {}
    qdr_by_q: dict[str, list[tuple[int, str]]] = {}
    for row in ref_rankings:
        ref_by_q.setdefault(row["query_id"], []).append(
            (row["rank"], row["long_unit_id"])
        )
    for row in qdr_rankings:
        qdr_by_q.setdefault(row["query_id"], []).append(
            (row["rank"], row["long_unit_id"])
        )
    assert set(ref_by_q) == set(qdr_by_q)
    for qid in ref_by_q:
        ref_sorted = sorted(ref_by_q[qid])
        qdr_sorted = sorted(qdr_by_q[qid])
        assert ref_sorted == qdr_sorted, f"rank order mismatch for {qid}"