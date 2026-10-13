"""Validate the pilot BGE-M3 execution manifest.

The manifest is written in PLANNED state before execution and is
updated in place by the materialization runner to SUCCEEDED state.
The validator accepts both states and enforces that the results
block is consistent with the declared status.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_embedding_execution_manifest.yaml")

FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

EXPECTED_ARTIFACT_TYPE = "pilot_embedding_execution_manifest"
EXPECTED_ARTIFACT_VERSION = "1.0"
EXPECTED_EXECUTION_ID = "pilot-bge-m3-embedding-v1"
EXPECTED_CONTRACT_VERSION = "0.1.0"
EXPECTED_MODEL_ID = "BAAI/bge-m3"
EXPECTED_NEXT_STEP = "qdrant_ingestion_plan"

ALLOWED_STATUSES = ("PLANNED", "SUCCEEDED", "INTERRUPTED")

FORBIDDEN_APPROVALS = (
    "model_download_approved",
    "qdrant_execution_approved",
    "retrieval_execution_approved",
    "evaluation_execution_approved",
    "generation_execution_approved",
)

RESULT_BOOLEAN_KEYS = (
    "model_loaded",
    "benchmark_performed",
    "embedding_materialization_performed",
    "round_trip_valid",
    "checksums_valid",
    "deterministic_data_artifacts",
)

RESULT_NULLABLE_INT_KEYS = (
    "dense_vector_count",
    "sparse_vector_count",
    "selected_batch_size",
    "output_bytes",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "manifest yaml must be a mapping")

    _require(data.get("artifact_type") == EXPECTED_ARTIFACT_TYPE,
             "artifact_type must be " + EXPECTED_ARTIFACT_TYPE)
    _require(data.get("artifact_version") == EXPECTED_ARTIFACT_VERSION,
             "artifact_version must be " + EXPECTED_ARTIFACT_VERSION)

    execution = data["execution"]
    _require(execution["execution_id"] == EXPECTED_EXECUTION_ID,
             "execution_id changed")
    _require(execution["contract_version"] == EXPECTED_CONTRACT_VERSION,
             "contract_version must be 0.1.0")
    status = execution["status"]
    _require(status in ALLOWED_STATUSES,
             "status must be one of " + ", ".join(ALLOWED_STATUSES))

    approval = data["approval"]
    _require(approval["embedding_execution_approved"] is True,
             "embedding_execution_approved must be true")
    for flag in FORBIDDEN_APPROVALS:
        _require(approval[flag] is False, "approval." + flag + " must be false")

    model = data["model"]
    _require(model["model_id"] == EXPECTED_MODEL_ID, "model_id must be BAAI/bge-m3")
    revision = model["model_revision"]
    _require(len(revision) == 40, "model_revision must be 40-char SHA")
    _require(revision == FROZEN_REVISION, "model_revision must equal frozen revision")
    _require(model["cache_status"] in ("VERIFIED", "MISSING"),
             "cache_status must be VERIFIED or MISSING")

    results = data["results"]
    for key in RESULT_BOOLEAN_KEYS:
        _require(key in results, "results missing " + key)
        _require(isinstance(results[key], bool), "results." + key + " must be bool")
    for key in RESULT_NULLABLE_INT_KEYS:
        _require(key in results, "results missing " + key)
        value = results[key]
        _require(value is None or (isinstance(value, int) and value >= 0),
                 "results." + key + " must be null or non-negative int")

    if status == "PLANNED":
        for key in RESULT_BOOLEAN_KEYS:
            _require(results[key] is False,
                     "PLANNED manifest results." + key + " must be false")
        for key in RESULT_NULLABLE_INT_KEYS:
            _require(results[key] is None,
                     "PLANNED manifest results." + key + " must be null")
    elif status == "SUCCEEDED":
        for key in ("model_loaded", "benchmark_performed",
                    "embedding_materialization_performed",
                    "round_trip_valid", "checksums_valid",
                    "deterministic_data_artifacts"):
            _require(results[key] is True,
                     "SUCCEEDED manifest results." + key + " must be true")
        _require(results["dense_vector_count"] == 12,
                 "SUCCEEDED manifest dense_vector_count must be 12")
        _require(results["sparse_vector_count"] == 12,
                 "SUCCEEDED manifest sparse_vector_count must be 12")
        _require(results["selected_batch_size"] in (1, 2, 4),
                 "SUCCEEDED selected_batch_size must be 1, 2 or 4")
        _require(isinstance(results["output_bytes"], int) and results["output_bytes"] > 0,
                 "SUCCEEDED output_bytes must be positive int")

    next_gate = data["next_gate"]
    _require(next_gate["required_step"] == EXPECTED_NEXT_STEP,
             "next_gate.required_step must be " + EXPECTED_NEXT_STEP)
    _require(next_gate["approved"] is False,
             "next_gate.approved must be false until Step 84")

    print("[OK] Pilot embedding execution manifest")
    print("  execution id:", execution["execution_id"])
    print("  contract version:", execution["contract_version"])
    print("  status:", status)
    print("  model:", model["model_id"])
    print("  revision:", revision)
    print("  cache status:", model["cache_status"])
    print("  embedding approved:", approval["embedding_execution_approved"])
    print("  model download approved:", approval["model_download_approved"])
    print("  next gate:", next_gate["required_step"],
          "(approved=" + str(next_gate["approved"]) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())