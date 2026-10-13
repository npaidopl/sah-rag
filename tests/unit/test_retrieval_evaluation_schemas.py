"""Tests for exported retrieval-evaluation schemas."""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_PATHS = {
    "case": Path(
        "schemas/retrieval_evaluation_case.schema.json"
    ),
    "metrics": Path(
        "schemas/retrieval_metrics.schema.json"
    ),
    "aggregates": Path(
        "schemas/retrieval_aggregate_metrics.schema.json"
    ),
    "run": Path(
        "schemas/retrieval_evaluation_run.schema.json"
    ),
}


def test_retrieval_evaluation_schemas_exist() -> None:
    """All retrieval-evaluation schemas must be exported."""
    for path in SCHEMA_PATHS.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_metrics_schema_requires_core_values() -> None:
    """Metric records must require their core measurements."""
    schema = json.loads(
        SCHEMA_PATHS["metrics"].read_text(
            encoding="utf-8"
        )
    )

    required = set(schema["required"])

    assert {
        "query_id",
        "retrieval_method",
        "evaluation_level",
        "cutoff",
        "recall_at_k",
        "precision_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
        "hit_at_k",
    }.issubset(required)


def test_run_schema_requires_artifact_paths() -> None:
    """Evaluation runs must retain output locations."""
    schema = json.loads(
        SCHEMA_PATHS["run"].read_text(
            encoding="utf-8"
        )
    )

    required = set(schema["required"])

    assert {
        "run_id",
        "dataset_id",
        "cases_path",
        "metrics_path",
        "aggregates_path",
    }.issubset(required)
