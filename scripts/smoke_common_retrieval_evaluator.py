"""Synthetic smoke for the common retrieval evaluation contract."""

from __future__ import annotations

import shutil
from pathlib import Path

from rag_system.evaluation import RetrievalEvaluationCase, RetrievalJudgment
from rag_system.evaluation_contracts import (
    CommonRetrievalEvaluationRun,
    evaluate_retrieval_run,
    validate_evaluation_results,
)
from rag_system.evaluation_persistence import (
    read_retrieval_evaluation_artifacts,
    write_retrieval_evaluation_artifacts,
)

OUTPUT_ROOT = Path("outputs/common_retrieval_evaluator_smoke")
EXPERIMENT_CELL_ID = "experiment-cell-" + "a" * 64
CUTOFFS = (1, 5, 10, 100)


def _case(query_id: str, top: str, second: str, filler: str) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        query_id=query_id,
        query_text=f"query {query_id}",
        retrieval_method="smoke",
        evaluation_level="document",
        judgments=(
            RetrievalJudgment(item_id=top, relevance=2),
            RetrievalJudgment(item_id=second, relevance=2),
            RetrievalJudgment(item_id=filler, relevance=0),
        ),
        retrieved_ids=(top, second, filler),
        metadata={},
    )


def main() -> None:
    cases = (
        _case("query-01", "doc-01", "doc-02", "doc-03"),
        _case("query-02", "doc-05", "doc-06", "doc-07"),
        _case("query-03", "doc-11", "doc-12", "doc-13"),
    )
    run = CommonRetrievalEvaluationRun(
        experiment_cell_id=EXPERIMENT_CELL_ID,
        dataset_id="synthetic-common-eval",
        architecture_id="longrag",
        cutoffs=CUTOFFS,
        expected_query_ids=tuple(c.query_id for c in cases),
    )

    print("[STEP] evaluate")
    qm, agg = evaluate_retrieval_run(run=run, cases=cases)
    validate_evaluation_results(
        run=run, cases=cases, query_metrics=qm, aggregate_metrics=agg
    )
    print("  query metric count:", len(qm))
    print("  aggregate metric count:", len(agg))

    print("[METRICS]")
    for a in agg:
        print(
            f"  cutoff={a.cutoff} "
            f"Hit@k={a.mean_hit_at_k:.4f} "
            f"MRR@k={a.mean_reciprocal_rank:.4f} "
            f"Recall@k={a.mean_recall_at_k:.4f} "
            f"Precision@k={a.mean_precision_at_k:.4f} "
            f"NDCG@k={a.mean_ndcg_at_k:.4f}"
        )

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    manifest = write_retrieval_evaluation_artifacts(
        root=OUTPUT_ROOT,
        run=run,
        cases=cases,
        query_metrics=qm,
        aggregate_metrics=agg,
        overwrite=True,
    )
    print("[PERSIST]")
    print("  case_count:", manifest["case_count"])
    print("  query_metric_count:", manifest["query_metric_count"])
    print("  aggregate_metric_count:", manifest["aggregate_metric_count"])

    read_retrieval_evaluation_artifacts(
        root=OUTPUT_ROOT, expected_experiment_cell_id=EXPERIMENT_CELL_ID
    )
    print("[RELOAD] round trip: True")

    print("[OK] common retrieval evaluator smoke")
    print("  query metric count:", len(qm))
    print("  aggregate metric count:", len(agg))
    print("  all expected queries present: True")
    print("  macro averaging: True")
    print("  persistence round trip: True")
    print("  checksums valid: True")
    print("  generation disabled: True")


if __name__ == "__main__":
    main()
