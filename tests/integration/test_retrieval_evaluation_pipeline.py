"""Integration tests for retrieval evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_system.artifacts import (
    ArtifactSidecarManifest,
    read_json_model,
    sha256_file,
)
from rag_system.canonical.jsonl import iter_jsonl
from rag_system.evaluation import (
    RetrievalAggregateMetrics,
    RetrievalEvaluationCase,
    RetrievalEvaluationRun,
    RetrievalJudgment,
    RetrievalMetrics,
    write_retrieval_evaluation,
)


def test_evaluation_pipeline_publishes_artifacts(
    tmp_path: Path,
) -> None:
    """Evaluation should publish typed, checksummed artifacts."""
    source_path = tmp_path / "predictions.jsonl"
    source_path.write_text(
        '{"query_id":"query-001"}\n',
        encoding="utf-8",
    )

    cases = (
        RetrievalEvaluationCase(
            query_id="query-001",
            query_text="first query",
            retrieval_method="dense",
            evaluation_level="chunk",
            judgments=(
                RetrievalJudgment(
                    item_id="chunk-a",
                    relevance=1,
                ),
            ),
            retrieved_ids=(
                "chunk-a",
                "chunk-b",
            ),
        ),
        RetrievalEvaluationCase(
            query_id="query-001",
            query_text="first query",
            retrieval_method="hybrid_rrf",
            evaluation_level="chunk",
            judgments=(
                RetrievalJudgment(
                    item_id="chunk-a",
                    relevance=1,
                ),
            ),
            retrieved_ids=(
                "chunk-a",
                "chunk-b",
            ),
        ),
    )

    summary = write_retrieval_evaluation(
        root=tmp_path / "evaluation",
        dataset_id="synthetic",
        evaluation_level="chunk",
        cases=cases,
        cutoffs=(1, 2),
        source_artifact_path=source_path,
        source_artifact_type="retrieval_predictions",
        execution_profile="test",
    )

    assert summary.case_count == 2
    assert summary.metric_record_count == 4
    assert summary.aggregate_record_count == 4

    run = read_json_model(
        summary.paths.run,
        RetrievalEvaluationRun,
    )

    manifest = read_json_model(
        summary.paths.manifest,
        ArtifactSidecarManifest,
    )

    metrics = [
        RetrievalMetrics.model_validate(record)
        for record in iter_jsonl(
            summary.paths.metrics
        )
    ]

    aggregates = [
        RetrievalAggregateMetrics.model_validate(
            record
        )
        for record in iter_jsonl(
            summary.paths.aggregates
        )
    ]

    assert run.case_count == 2
    assert len(metrics) == 4
    assert len(aggregates) == 4

    assert manifest.record_count == 4
    assert manifest.artifact_sha256 == sha256_file(
        summary.paths.metrics
    )


def test_evaluation_pipeline_removes_partial_outputs(
    tmp_path: Path,
) -> None:
    """Invalid mixed levels should publish no artifacts."""
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        '{"id":"source"}\n',
        encoding="utf-8",
    )

    cases = (
        RetrievalEvaluationCase(
            query_id="query-001",
            query_text="query",
            retrieval_method="dense",
            evaluation_level="parent",
            judgments=(),
            retrieved_ids=(),
        ),
    )

    root = tmp_path / "evaluation"

    with pytest.raises(
        ValueError,
        match="level does not match",
    ):
        write_retrieval_evaluation(
            root=root,
            dataset_id="synthetic",
            evaluation_level="chunk",
            cases=cases,
            cutoffs=(1,),
            source_artifact_path=source_path,
            source_artifact_type="predictions",
        )

    if root.exists():
        assert not any(root.iterdir())
