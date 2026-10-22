"""Tests for pilot evaluation persistence (Step 89)."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation import RetrievalEvaluationCase
from evaluation.models import RetrievalJudgment
from evaluation.pilot_persistence import (
    read_pilot_evaluation_artifacts,
    write_pilot_evaluation_artifacts,
)
from evaluation.contracts import (
    AggregateCutoffMetrics,
    QueryCutoffMetrics,
)


def _case(qid: str) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        query_id=qid,
        query_text=f"text-{qid}",
        retrieval_method="hybrid_rrf",
        evaluation_level="document",
        judgments=(
            RetrievalJudgment(item_id=f"doc-{qid}-1", relevance=1),
            RetrievalJudgment(item_id=f"doc-{qid}-2", relevance=1),
        ),
        retrieved_ids=(f"doc-{qid}-1", f"doc-{qid}-2"),
        metadata={},
    )


def _qm(qid: str, cutoff: int, base: float) -> QueryCutoffMetrics:
    return QueryCutoffMetrics(
        experiment_cell_id="e1",
        dataset_id="d1",
        architecture_id="sah_rag",
        query_id=qid,
        cutoff=cutoff,
        hit_at_k=base,
        precision_at_k=base,
        recall_at_k=base,
        reciprocal_rank=base,
        ndcg_at_k=base,
        relevant_item_count=2,
        retrieved_item_count=2,
        retrieved_within_cutoff_count=min(cutoff, 2),
    )


def _agg(cutoff: int, mean: float) -> AggregateCutoffMetrics:
    return AggregateCutoffMetrics(
        experiment_cell_id="e1",
        dataset_id="d1",
        architecture_id="sah_rag",
        cutoff=cutoff,
        query_count=6,
        mean_hit_at_k=mean,
        mean_precision_at_k=mean,
        mean_recall_at_k=mean,
        mean_reciprocal_rank=mean,
        mean_ndcg_at_k=mean,
    )


def _write(tmp_path: Path) -> Path:
    root = tmp_path / "evaluation"
    cases = [_case(f"q-{i}") for i in range(1, 7)]
    qms = [_qm(f"q-{i}", c, 0.5) for i in range(1, 7) for c in (1, 5, 10, 100)]
    aggs = [_agg(c, 0.5) for c in (1, 5, 10, 100)]
    write_pilot_evaluation_artifacts(
        root=root,
        retrieval_cases=cases,
        query_metrics=qms,
        aggregate_metrics=aggs,
        manifest={"artifact_type": "pilot_retrieval_evaluation_manifest"},
    )
    return root


def test_round_trip_succeeds(tmp_path: Path):
    root = _write(tmp_path)
    loaded = read_pilot_evaluation_artifacts(root=root, expected_query_count=6)
    assert len(loaded["retrieval_cases"]) == 6
    assert loaded["checksums_valid"] is True


def test_checksums_validate(tmp_path: Path):
    root = _write(tmp_path)
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    assert checksums["algorithm"] == "sha256"
    assert "query_metrics.jsonl" in checksums["files"]


def test_existing_root_rejected_without_overwrite(tmp_path: Path):
    root = _write(tmp_path)
    try:
        write_pilot_evaluation_artifacts(
            root=root,
            retrieval_cases=[],
            query_metrics=[],
            aggregate_metrics=[],
            manifest={},
        )
    except FileExistsError:
        return
    raise AssertionError("expected FileExistsError")


def test_repeat_run_byte_identical(tmp_path: Path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    cases = [_case(f"q-{i}") for i in range(1, 7)]
    qms = [_qm(f"q-{i}", c, 0.5) for i in range(1, 7) for c in (1, 5, 10, 100)]
    aggs = [_agg(c, 0.5) for c in (1, 5, 10, 100)]
    for root in (root_a, root_b):
        write_pilot_evaluation_artifacts(
            root=root,
            retrieval_cases=cases,
            query_metrics=qms,
            aggregate_metrics=aggs,
            manifest={"artifact_type": "pilot_retrieval_evaluation_manifest"},
        )
    for name in (
        "retrieval_cases.jsonl",
        "query_metrics.jsonl",
        "aggregate_metrics.json",
    ):
        assert (root_a / name).read_bytes() == (root_b / name).read_bytes()