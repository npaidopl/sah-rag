"""Writing and provenance of retrieval evaluation artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_system.artifacts import (
    ArtifactComponentReference,
    build_input_reference,
    build_jsonl_sidecar_manifest,
    get_git_revision,
    sha256_fingerprint,
    write_json_model_atomic,
)
from rag_system.artifacts.jsonl import write_jsonl_atomic
from rag_system.evaluation.identity import (
    build_evaluation_run_id,
)
from rag_system.evaluation.metrics import (
    aggregate_metrics,
    evaluate_case,
)
from rag_system.evaluation.models import (
    RetrievalEvaluationCase,
    RetrievalEvaluationRun,
    RetrievalMetrics,
)


@dataclass(frozen=True, slots=True)
class EvaluationArtifactPaths:
    """Files produced by one retrieval evaluation."""

    root: Path
    cases: Path
    metrics: Path
    aggregates: Path
    run: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class EvaluationWriteSummary:
    """Summary of a completed evaluation artifact run."""

    run_id: str
    case_count: int
    metric_record_count: int
    aggregate_record_count: int
    paths: EvaluationArtifactPaths


def resolve_evaluation_paths(
    root: Path,
) -> EvaluationArtifactPaths:
    """Resolve conventional evaluation artifact paths."""
    return EvaluationArtifactPaths(
        root=root,
        cases=root / "cases.jsonl",
        metrics=root / "metrics.jsonl",
        aggregates=root / "aggregates.jsonl",
        run=root / "run.json",
        manifest=root / "evaluation.manifest.json",
    )


def write_retrieval_evaluation(
    *,
    root: Path,
    dataset_id: str,
    evaluation_level: str,
    cases: Iterable[RetrievalEvaluationCase],
    cutoffs: tuple[int, ...],
    source_artifact_path: Path,
    source_artifact_type: str,
    source_manifest_path: Path | None = None,
    overwrite: bool = False,
    execution_profile: str | None = None,
    experiment_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationWriteSummary:
    """Evaluate cases and atomically publish evaluation artifacts."""
    normalized_cutoffs = tuple(
        sorted(set(cutoffs))
    )

    if not normalized_cutoffs:
        raise ValueError(
            "At least one evaluation cutoff is required."
        )

    if any(cutoff < 1 for cutoff in normalized_cutoffs):
        raise ValueError(
            "Evaluation cutoffs must be positive."
        )

    case_records = tuple(cases)

    if not case_records:
        raise ValueError(
            "At least one evaluation case is required."
        )

    for case in case_records:
        if case.evaluation_level != evaluation_level:
            raise ValueError(
                "Evaluation case level does not match the run level."
            )

    retrieval_methods = tuple(
        sorted({
            case.retrieval_method
            for case in case_records
        })
    )

    configuration = {
        "dataset_id": dataset_id,
        "evaluation_level": evaluation_level,
        "retrieval_methods": retrieval_methods,
        "cutoffs": normalized_cutoffs,
        "metric_definitions": {
            "recall": "binary_recall_at_k",
            "precision": "fixed_cutoff_precision_at_k",
            "reciprocal_rank": "first_relevant_reciprocal_rank",
            "ndcg": "graded_exponential_gain_ndcg",
            "hit": "binary_hit_at_k",
            "aggregation": "macro_average",
        },
    }

    configuration_hash = sha256_fingerprint(
        configuration
    )

    run_id = build_evaluation_run_id(
        dataset_id=dataset_id,
        evaluation_level=evaluation_level,
        retrieval_methods=retrieval_methods,
        cutoffs=normalized_cutoffs,
        configuration_hash=configuration_hash,
    )

    paths = resolve_evaluation_paths(root)

    paths.root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not overwrite:
        existing = [
            path
            for path in (
                paths.cases,
                paths.metrics,
                paths.aggregates,
                paths.run,
                paths.manifest,
            )
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "Evaluation outputs already exist: "
                + ", ".join(
                    str(path)
                    for path in existing
                )
            )

    try:
        metric_records = tuple(
            evaluate_case(
                case,
                cutoff=cutoff,
            )
            for case in case_records
            for cutoff in normalized_cutoffs
        )

        grouped: dict[
            tuple[str, str, int],
            list[RetrievalMetrics],
        ] = defaultdict(list)

        for metric in metric_records:
            key = (
                metric.retrieval_method,
                metric.evaluation_level,
                metric.cutoff,
            )
            grouped[key].append(metric)

        aggregate_records = tuple(
            aggregate_metrics(group)
            for _, group in sorted(
                grouped.items(),
                key=lambda item: item[0],
            )
        )

        write_jsonl_atomic(
            paths.cases,
            case_records,
            overwrite=overwrite,
        )

        write_jsonl_atomic(
            paths.metrics,
            metric_records,
            overwrite=overwrite,
        )

        write_jsonl_atomic(
            paths.aggregates,
            aggregate_records,
            overwrite=overwrite,
        )

        run = RetrievalEvaluationRun(
            run_id=run_id,
            dataset_id=dataset_id,
            evaluation_level=evaluation_level,
            retrieval_methods=retrieval_methods,
            cutoffs=normalized_cutoffs,
            case_count=len(case_records),
            metric_record_count=len(metric_records),
            aggregate_record_count=len(
                aggregate_records
            ),
            cases_path=str(paths.cases.resolve()),
            metrics_path=str(paths.metrics.resolve()),
            aggregates_path=str(
                paths.aggregates.resolve()
            ),
            metadata=(
                metadata
                if metadata is not None
                else {}
            ),
        )

        write_json_model_atomic(
            paths.run,
            run,
            overwrite=overwrite,
        )

        source_reference = build_input_reference(
            artifact_type=source_artifact_type,
            path=source_artifact_path,
        )

        inputs = [source_reference]

        if source_manifest_path is not None:
            source_manifest_reference = (
                build_input_reference(
                    artifact_type=(
                        f"{source_artifact_type}_manifest"
                    ),
                    path=source_manifest_path,
                    record_count=1,
                )
            )
            inputs.append(source_manifest_reference)

        run_reference = build_input_reference(
            artifact_type="retrieval_evaluation_run",
            path=paths.run,
            record_count=1,
        )

        cases_reference = build_input_reference(
            artifact_type="retrieval_evaluation_cases",
            path=paths.cases,
            record_count=len(case_records),
        )

        aggregates_reference = build_input_reference(
            artifact_type="retrieval_aggregate_metrics",
            path=paths.aggregates,
            record_count=len(aggregate_records),
        )

        inputs.extend(
            [
                run_reference,
                cases_reference,
                aggregates_reference,
            ]
        )

        component = ArtifactComponentReference(
            component_type="retrieval_evaluator",
            component_id="standard_retrieval_metrics",
            component_version="0.1.0",
            configuration_hash=configuration_hash,
            configuration=configuration,
        )

        manifest_metadata: dict[str, Any] = {
            "run_id": run_id,
            "case_count": len(case_records),
            "metric_record_count": len(
                metric_records
            ),
            "aggregate_record_count": len(
                aggregate_records
            ),
        }

        if metadata is not None:
            manifest_metadata.update(metadata)

        manifest = build_jsonl_sidecar_manifest(
            artifact_type="retrieval_metrics",
            artifact_path=paths.metrics,
            component=component,
            dataset_id=dataset_id,
            experiment_id=experiment_id,
            inputs=tuple(inputs),
            code_revision=get_git_revision(),
            execution_profile=execution_profile,
            metadata=manifest_metadata,
        )

        if manifest.record_count != len(
            metric_records
        ):
            raise RuntimeError(
                "Evaluation manifest count does not match metrics."
            )

        write_json_model_atomic(
            paths.manifest,
            manifest,
            overwrite=overwrite,
        )

    except BaseException:
        for path in (
            paths.cases,
            paths.metrics,
            paths.aggregates,
            paths.run,
            paths.manifest,
        ):
            path.unlink(missing_ok=True)
        raise

    return EvaluationWriteSummary(
        run_id=run_id,
        case_count=len(case_records),
        metric_record_count=len(metric_records),
        aggregate_record_count=len(
            aggregate_records
        ),
        paths=paths,
    )
