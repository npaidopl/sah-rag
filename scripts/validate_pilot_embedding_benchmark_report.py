"""Validate the pilot BGE-M3 CPU benchmark report."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_embedding_benchmark_report.yaml")

EXPECTED_ARTIFACT_TYPE = "pilot_embedding_benchmark_report"
EXPECTED_ARTIFACT_VERSION = "1.0"
EXPECTED_BENCHMARK_ID = "bge_m3_cpu_pilot_benchmark"
EXPECTED_PROTOCOL_VERSION = "0.1.0"
EXPECTED_MODEL_ID = "BAAI/bge-m3"
FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

ALLOWED_STATUSES = ("PLANNED", "SUCCEEDED", "INTERRUPTED")

FORBIDDEN_FLAGS = (
    "model_download_performed",
    "network_access_performed",
    "qdrant_execution_performed",
    "retrieval_execution_performed",
    "evaluation_execution_performed",
    "generation_execution_performed",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _validate_ok_batch(entry: dict) -> None:
    _require(entry.get("status") == "OK", "batch entry status must be OK")
    runs = entry.get("raw_embedding_seconds")
    _require(isinstance(runs, list) and len(runs) == 3,
             "each batch must retain exactly three raw runs")
    for run in runs:
        _require(isinstance(run, (int, float)) and run > 0,
                 "raw run must be a positive number")
    median_seconds = entry.get("median_seconds")
    _require(isinstance(median_seconds, (int, float)) and median_seconds > 0,
             "median_seconds must be positive")
    _require(entry.get("minimum_seconds") == min(runs),
             "minimum_seconds must match raw runs")
    _require(entry.get("maximum_seconds") == max(runs),
             "maximum_seconds must match raw runs")
    _require(entry.get("median_records_per_second") > 0,
             "median_records_per_second must be positive")
    _require(entry.get("median_milliseconds_per_record") > 0,
             "median_milliseconds_per_record must be positive")
    _require(isinstance(entry.get("peak_process_ram_bytes"), int),
             "peak_process_ram_bytes must be int")


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "report must be a mapping")

    _require(data.get("artifact_type") == EXPECTED_ARTIFACT_TYPE,
             "artifact_type must be " + EXPECTED_ARTIFACT_TYPE)
    _require(data.get("artifact_version") == EXPECTED_ARTIFACT_VERSION,
             "artifact_version must be " + EXPECTED_ARTIFACT_VERSION)

    benchmark = data["benchmark"]
    _require(benchmark["id"] == EXPECTED_BENCHMARK_ID, "benchmark.id changed")
    _require(benchmark["protocol_version"] == EXPECTED_PROTOCOL_VERSION,
             "protocol_version must be 0.1.0")
    status = benchmark["status"]
    _require(status in ALLOWED_STATUSES,
             "status must be one of " + ", ".join(ALLOWED_STATUSES))

    model = data["model"]
    _require(model["model_id"] == EXPECTED_MODEL_ID, "model_id must be BAAI/bge-m3")
    _require(model["model_revision"] == FROZEN_REVISION,
             "model_revision must equal frozen revision")
    _require(model["device"] == "cpu", "device must be cpu")
    _require(model["local_files_only"] is True, "local_files_only must be true")

    for flag in FORBIDDEN_FLAGS:
        _require(data[flag] is False, flag + " must be false")

    interpretation = data["interpretation"]
    _require(interpretation["publishable"] is False, "publishable must be false")
    _require(interpretation["real_dataset_estimate"] is False,
             "real_dataset_estimate must be false")
    _require(interpretation["linear_extrapolation_allowed"] is False,
             "linear_extrapolation_allowed must be false")

    execution_state = data["execution_state"]
    _require(execution_state["embeddings_persisted"] is False,
             "embeddings_persisted must be false during benchmark")

    if status == "PLANNED":
        _require(data["results"] == {}, "PLANNED report results must be empty")
        _require(data["selection"]["selected_batch_size"] is None,
                 "PLANNED selection.selected_batch_size must be null")
        _require(execution_state["model_loaded"] is False,
                 "PLANNED model_loaded must be false")
    elif status == "SUCCEEDED":
        results = data["results"]
        for bs in (1, 2, 4):
            key = "batch_size_" + str(bs)
            _require(key in results, "SUCCEEDED results missing " + key)
            _validate_ok_batch(results[key])
        _require(data["selection"]["selected_batch_size"] in (1, 2, 4),
                 "selected_batch_size must be 1, 2 or 4")
        _require(execution_state["model_loaded"] is True,
                 "SUCCEEDED model_loaded must be true")
        _require(execution_state["embeddings_computed_for_benchmark"] is True,
                 "SUCCEEDED embeddings_computed_for_benchmark must be true")
        _require(data["input"]["record_count"] == 12,
                 "SUCCEEDED record_count must be 12")
    elif status == "INTERRUPTED":
        _require("failure_stage" in data,
                 "INTERRUPTED report must declare failure_stage")

    print("[OK] Pilot BGE-M3 CPU benchmark report")
    print("  benchmark id:", benchmark["id"])
    print("  status:", status)
    print("  model:", model["model_id"])
    print("  revision:", model["model_revision"])
    print("  selected batch size:", data["selection"]["selected_batch_size"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())