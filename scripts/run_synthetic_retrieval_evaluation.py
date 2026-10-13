"""Create a synthetic multi-method retrieval evaluation."""

from __future__ import annotations

from pathlib import Path

from rag_system.evaluation import (
    RetrievalEvaluationCase,
    RetrievalJudgment,
    write_retrieval_evaluation,
)

OUTPUT_ROOT = Path(
    "outputs/synthetic_retrieval_evaluation"
)

SOURCE_PATH = Path(
    "outputs/clerk_chunks_auto.jsonl"
)

SOURCE_MANIFEST_PATH = Path(
    "outputs/clerk_chunks_auto.manifest.json"
)


def main() -> None:
    """Run a deterministic infrastructure-only evaluation."""
    judgments = (
        RetrievalJudgment(
            item_id="chunk-relevant",
            relevance=2,
        ),
        RetrievalJudgment(
            item_id="chunk-partial",
            relevance=1,
        ),
    )

    cases = (
        RetrievalEvaluationCase(
            query_id="query-001",
            query_text="synthetic evaluation query",
            retrieval_method="dense",
            evaluation_level="chunk",
            judgments=judgments,
            retrieved_ids=(
                "chunk-noise",
                "chunk-relevant",
                "chunk-partial",
            ),
            metadata={
                "synthetic": True,
            },
        ),
        RetrievalEvaluationCase(
            query_id="query-001",
            query_text="synthetic evaluation query",
            retrieval_method="sparse",
            evaluation_level="chunk",
            judgments=judgments,
            retrieved_ids=(
                "chunk-partial",
                "chunk-noise",
                "chunk-relevant",
            ),
            metadata={
                "synthetic": True,
            },
        ),
        RetrievalEvaluationCase(
            query_id="query-001",
            query_text="synthetic evaluation query",
            retrieval_method="hybrid_rrf",
            evaluation_level="chunk",
            judgments=judgments,
            retrieved_ids=(
                "chunk-relevant",
                "chunk-partial",
                "chunk-noise",
            ),
            metadata={
                "synthetic": True,
            },
        ),
    )

    summary = write_retrieval_evaluation(
        root=OUTPUT_ROOT,
        dataset_id="synthetic",
        evaluation_level="chunk",
        cases=cases,
        cutoffs=(1, 3),
        source_artifact_path=SOURCE_PATH,
        source_artifact_type="child_chunks",
        source_manifest_path=SOURCE_MANIFEST_PATH,
        overwrite=True,
        execution_profile="local_cpu",
        metadata={
            "synthetic": True,
            "scientific_result": False,
        },
    )

    print("[OK] Synthetic retrieval evaluation")
    print(" Run ID:", summary.run_id)
    print(" Cases:", summary.case_count)
    print(
        " Metric records:",
        summary.metric_record_count,
    )
    print(
        " Aggregate records:",
        summary.aggregate_record_count,
    )
    print(" Cases file:", summary.paths.cases.resolve())
    print(
        " Metrics file:",
        summary.paths.metrics.resolve(),
    )
    print(
        " Aggregates file:",
        summary.paths.aggregates.resolve(),
    )
    print(
        " Manifest:",
        summary.paths.manifest.resolve(),
    )
    print(
        " WARNING: synthetic evaluation, "
        "not a scientific benchmark result."
    )


if __name__ == "__main__":
    main()
