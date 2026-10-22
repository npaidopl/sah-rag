"""Deterministic persistence for the common retrieval evaluation contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sah_rag.artifacts import sha256_file
from evaluation import RetrievalEvaluationCase
from evaluation.contracts import (
    AggregateCutoffMetrics,
    CommonRetrievalEvaluationManifest,
    CommonRetrievalEvaluationRun,
    QueryCutoffMetrics,
)


def _atomic_write_jsonl(path: Path, models: tuple[BaseModel, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = "".join(
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for model in models
    )
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _read_jsonl_models(path: Path, model_type: type[BaseModel]) -> tuple[BaseModel, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL artifact not found: {path}")
    return tuple(
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def write_retrieval_evaluation_artifacts(
    *,
    root: Path,
    run: CommonRetrievalEvaluationRun,
    cases: tuple[RetrievalEvaluationCase, ...],
    query_metrics: tuple[QueryCutoffMetrics, ...],
    aggregate_metrics: tuple[AggregateCutoffMetrics, ...],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the four evaluation artifacts plus their manifest."""
    cases_path = root / "retrieval_cases.jsonl"
    query_metrics_path = root / "query_metrics.jsonl"
    aggregate_metrics_path = root / "aggregate_metrics.json"
    manifest_path = root / "evaluation.manifest.json"

    paths = (cases_path, query_metrics_path, aggregate_metrics_path, manifest_path)
    if not overwrite and any(p.exists() for p in paths):
        raise FileExistsError("evaluation artifacts already exist")

    root.mkdir(parents=True, exist_ok=True)

    ordered_cases = tuple(sorted(cases, key=lambda c: c.query_id))
    ordered_qm = tuple(
        sorted(query_metrics, key=lambda m: (m.query_id, m.cutoff))
    )
    ordered_agg = tuple(sorted(aggregate_metrics, key=lambda a: a.cutoff))

    _atomic_write_jsonl(cases_path, ordered_cases)
    _atomic_write_jsonl(query_metrics_path, ordered_qm)
    _atomic_write_json(aggregate_metrics_path, [a.model_dump(mode="json") for a in ordered_agg])

    manifest = CommonRetrievalEvaluationManifest(
        experiment_cell_id=run.experiment_cell_id,
        dataset_id=run.dataset_id,
        architecture_id=run.architecture_id,
        query_count=len(run.expected_query_ids),
        cutoffs=tuple(run.cutoffs),
        case_count=len(ordered_cases),
        query_metric_count=len(ordered_qm),
        aggregate_metric_count=len(ordered_agg),
        cases_path="retrieval_cases.jsonl",
        query_metrics_path="query_metrics.jsonl",
        aggregate_metrics_path="aggregate_metrics.json",
        cases_sha256=sha256_file(cases_path),
        query_metrics_sha256=sha256_file(query_metrics_path),
        aggregate_metrics_sha256=sha256_file(aggregate_metrics_path),
        retrieval_only=True,
        final_generation_enabled=False,
        metadata={},
    )
    _atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    return manifest.model_dump(mode="json")


def read_retrieval_evaluation_artifacts(
    *,
    root: Path,
    expected_experiment_cell_id: str | None = None,
) -> dict[str, Any]:
    """Read, checksum-verify, and cross-validate the evaluation artifact set."""
    manifest_path = root / "evaluation.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"evaluation manifest not found: {manifest_path}")
    manifest = CommonRetrievalEvaluationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    cases_path = root / manifest.cases_path
    query_metrics_path = root / manifest.query_metrics_path
    aggregate_metrics_path = root / manifest.aggregate_metrics_path

    for expected_sha, path in (
        (manifest.cases_sha256, cases_path),
        (manifest.query_metrics_sha256, query_metrics_path),
        (manifest.aggregate_metrics_sha256, aggregate_metrics_path),
    ):
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"checksum mismatch: {path.name}")

    if (
        expected_experiment_cell_id is not None
        and manifest.experiment_cell_id != expected_experiment_cell_id
    ):
        raise RuntimeError("experiment_cell_id mismatch")

    cases = _read_jsonl_models(cases_path, RetrievalEvaluationCase)
    query_metrics = _read_jsonl_models(query_metrics_path, QueryCutoffMetrics)
    aggregate_payload = json.loads(
        aggregate_metrics_path.read_text(encoding="utf-8")
    )
    aggregate_metrics = tuple(
        AggregateCutoffMetrics.model_validate(x) for x in aggregate_payload
    )

    if len(cases) != manifest.case_count:
        raise RuntimeError("case_count mismatch")
    if len(query_metrics) != manifest.query_metric_count:
        raise RuntimeError("query_metric_count mismatch")
    if len(aggregate_metrics) != manifest.aggregate_metric_count:
        raise RuntimeError("aggregate_metric_count mismatch")

    return {
        "cases": cases,
        "query_metrics": query_metrics,
        "aggregate_metrics": aggregate_metrics,
        "manifest": manifest,
    }
