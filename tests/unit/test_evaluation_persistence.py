"""Tests for the common retrieval evaluation persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation import RetrievalEvaluationCase, RetrievalJudgment
from evaluation.contracts import (
    CommonRetrievalEvaluationRun,
    evaluate_retrieval_run,
)
from evaluation.persistence import (
    read_retrieval_evaluation_artifacts,
    write_retrieval_evaluation_artifacts,
)


def _case(query_id: str, retrieved: tuple[str, ...], judgments) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        query_id=query_id,
        query_text=query_id,
        retrieval_method="unit",
        evaluation_level="document",
        judgments=tuple(RetrievalJudgment(item_id=i, relevance=r) for i, r in judgments),
        retrieved_ids=retrieved,
        metadata={},
    )


def _fixture():
    run = CommonRetrievalEvaluationRun(
        experiment_cell_id="experiment-cell-" + "a" * 64,
        dataset_id="synthetic",
        architecture_id="longrag",
        cutoffs=(1, 5),
        expected_query_ids=("q-1",),
    )
    cases = (_case("q-1", ("a", "b"), [("a", 1), ("b", 0)]),)
    qm, agg = evaluate_retrieval_run(run=run, cases=cases)
    return run, cases, qm, agg


def test_round_trip_succeeds(tmp_path: Path):
    run, cases, qm, agg = _fixture()
    write_retrieval_evaluation_artifacts(
        root=tmp_path, run=run, cases=cases, query_metrics=qm, aggregate_metrics=agg,
        overwrite=True,
    )
    loaded = read_retrieval_evaluation_artifacts(
        root=tmp_path, expected_experiment_cell_id=run.experiment_cell_id,
    )
    assert len(loaded["cases"]) == len(cases)
    assert len(loaded["query_metrics"]) == len(qm)
    assert len(loaded["aggregate_metrics"]) == len(agg)


def test_existing_artifacts_rejected_without_overwrite(tmp_path: Path):
    run, cases, qm, agg = _fixture()
    write_retrieval_evaluation_artifacts(
        root=tmp_path, run=run, cases=cases, query_metrics=qm, aggregate_metrics=agg,
        overwrite=True,
    )
    with pytest.raises(FileExistsError):
        write_retrieval_evaluation_artifacts(
            root=tmp_path, run=run, cases=cases, query_metrics=qm, aggregate_metrics=agg,
            overwrite=False,
        )


def test_checksum_mismatch_rejected(tmp_path: Path):
    run, cases, qm, agg = _fixture()
    write_retrieval_evaluation_artifacts(
        root=tmp_path, run=run, cases=cases, query_metrics=qm, aggregate_metrics=agg,
        overwrite=True,
    )
    p = tmp_path / "query_metrics.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        read_retrieval_evaluation_artifacts(root=tmp_path)


def test_experiment_cell_mismatch_rejected(tmp_path: Path):
    run, cases, qm, agg = _fixture()
    write_retrieval_evaluation_artifacts(
        root=tmp_path, run=run, cases=cases, query_metrics=qm, aggregate_metrics=agg,
        overwrite=True,
    )
    with pytest.raises(RuntimeError, match="experiment_cell_id mismatch"):
        read_retrieval_evaluation_artifacts(
            root=tmp_path, expected_experiment_cell_id="experiment-cell-" + "b" * 64,
        )


def test_deterministic_output_bytes(tmp_path: Path):
    run, cases, qm, agg = _fixture()
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for root in (root_a, root_b):
        write_retrieval_evaluation_artifacts(
            root=root, run=run, cases=cases, query_metrics=qm,
            aggregate_metrics=agg, overwrite=True,
        )
    for name in (
        "retrieval_cases.jsonl",
        "query_metrics.jsonl",
        "aggregate_metrics.json",
    ):
        assert (root_a / name).read_bytes() == (root_b / name).read_bytes()
