"""Validate the pilot BGE-M3 execution contract for Step 83."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_embedding_execution_contract.yaml")

FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

EXPECTED_ARTIFACT_TYPE = "pilot_embedding_execution_contract"
EXPECTED_ARTIFACT_VERSION = "1.0"
EXPECTED_CONTRACT_ID = "infrastructure_pilot_bge_m3_cpu_execution"
EXPECTED_CONTRACT_VERSION = "0.1.0"
EXPECTED_CONTRACT_STATUS = "APPROVED_FOR_BOUNDED_EXECUTION"

EXPECTED_DATASET_ID = "infrastructure-pilot-synthetic-v1"
EXPECTED_CHILD_CHUNK_COUNT = 12

EXPECTED_MODEL_ID = "BAAI/bge-m3"
EXPECTED_DENSE_DIMENSION = 1024
EXPECTED_BATCH_SIZES = [1, 2, 4]

FORBIDDEN_EXECUTION_FLAGS = (
    "model_download_enabled",
    "network_access_enabled",
    "hosted_requests_enabled",
    "qdrant_collection_creation_enabled",
    "vector_ingestion_enabled",
    "retrieval_enabled",
    "evaluation_enabled",
    "generation_enabled",
)

FORBIDDEN_ACCEPTANCE_GATES = (
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


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "contract yaml must be a mapping")

    _require(data.get("artifact_type") == EXPECTED_ARTIFACT_TYPE,
             "artifact_type must be " + EXPECTED_ARTIFACT_TYPE)
    _require(data.get("artifact_version") == EXPECTED_ARTIFACT_VERSION,
             "artifact_version must be " + EXPECTED_ARTIFACT_VERSION)

    contract = data["contract"]
    _require(contract["id"] == EXPECTED_CONTRACT_ID, "contract.id changed")
    _require(contract["version"] == EXPECTED_CONTRACT_VERSION, "contract.version changed")
    _require(contract["status"] == EXPECTED_CONTRACT_STATUS, "contract.status changed")

    source = data["source"]
    _require(source["dataset_id"] == EXPECTED_DATASET_ID, "dataset_id changed")
    _require(source["expected_child_chunk_count"] == EXPECTED_CHILD_CHUNK_COUNT,
             "expected_child_chunk_count must be 12")

    model = data["model"]
    _require(model["model_id"] == EXPECTED_MODEL_ID, "model_id must be BAAI/bge-m3")
    revision = model["model_revision"]
    _require(len(revision) == 40, "model_revision must be 40-char SHA")
    _require(revision == FROZEN_REVISION, "model_revision must equal frozen revision")
    _require(model["revision_pinned"] is True, "revision_pinned must be true")
    _require(model["license_spdx"] == "MIT", "license_spdx must be MIT")
    _require(model["local_files_only"] is True, "local_files_only must be true")
    _require(model["network_fallback_allowed"] is False, "network_fallback_allowed must be false")

    init_params = model["init_params"]
    _require(init_params["use_fp16"] is False, "init_params.use_fp16 must be false on CPU")
    _require(init_params["use_bf16"] is False, "init_params.use_bf16 must be false on CPU")
    _require(init_params["pooling_method"] == "cls", "pooling_method must be cls")
    _require(init_params["normalize_embeddings"] is True, "normalize_embeddings must be true")
    _require(init_params["return_dense"] is True, "return_dense must be true")
    _require(init_params["return_sparse"] is True, "return_sparse must be true")
    _require(init_params["return_colbert_vecs"] is False, "return_colbert_vecs must be false")
    _require(init_params["devices"] == "cpu", "init_params.devices must be cpu")

    _require(model["dense_enabled"] is True, "dense_enabled must be true")
    _require(model["sparse_enabled"] is True, "sparse_enabled must be true")
    _require(model["multi_vector_enabled"] is False, "multi_vector_enabled must be false")
    _require(model["dense_dimension"] == EXPECTED_DENSE_DIMENSION,
             "dense_dimension must be 1024")
    _require(model["dense_normalized"] is True, "dense_normalized must be true")

    execution = data["execution"]
    _require(execution["model_load_enabled"] is True, "model_load_enabled must be true")
    _require(execution["embedding_enabled"] is True, "embedding_enabled must be true")
    for flag in FORBIDDEN_EXECUTION_FLAGS:
        _require(execution[flag] is False, "execution." + flag + " must be false")
    _require(execution["device"] == "cpu", "device must be cpu")
    _require(execution["worker_count"] == 1, "worker_count must be 1")
    _require(execution["multiprocessing_enabled"] is False,
             "multiprocessing_enabled must be false")

    benchmark = data["benchmark"]
    _require(benchmark["batch_sizes"] == EXPECTED_BATCH_SIZES,
             "batch_sizes must be [1, 2, 4]")
    _require(benchmark["warmup_run_count"] == 1, "warmup_run_count must be 1")
    _require(benchmark["measured_run_count_per_batch_size"] == 3,
             "measured_run_count_per_batch_size must be 3")
    _require(benchmark["raw_runs_retained"] is True, "raw_runs_retained must be true")
    _require(benchmark["linear_extrapolation_allowed"] is False,
             "linear_extrapolation_allowed must be false")

    materialization = data["materialization"]
    _require(materialization["expected_shard_count"] == 1,
             "expected_shard_count must be 1")
    _require(materialization["records_per_shard"] == EXPECTED_CHILD_CHUNK_COUNT,
             "records_per_shard must be 12")
    required = materialization["required_files"]
    for name in (
        "shards/shard-00000/header.json",
        "shards/shard-00000/records.jsonl",
        "shards/shard-00000/shard.manifest.json",
        "embedding.catalog.json",
        "embedding.manifest.json",
        "checksums.json",
    ):
        _require(name in required, "required_files missing " + name)
    _require(materialization.get("shards_subdir") == "shards",
             "shards_subdir must be shards")

    limits = data["resource_limits"]
    _require(limits["minimum_available_ram_bytes"] >= 2147483648,
             "minimum_available_ram_bytes must be >= 2 GiB")
    _require(limits["maximum_wall_clock_seconds"] > 0,
             "maximum_wall_clock_seconds must be positive")

    offline = data["offline_enforcement"]
    _require(offline["HF_HUB_OFFLINE"] == "1", "HF_HUB_OFFLINE must be 1")
    _require(offline["TRANSFORMERS_OFFLINE"] == "1", "TRANSFORMERS_OFFLINE must be 1")
    _require(offline["HF_DATASETS_OFFLINE"] == "1", "HF_DATASETS_OFFLINE must be 1")
    _require(offline["CUDA_VISIBLE_DEVICES"] == "", "CUDA_VISIBLE_DEVICES must be empty")
    _require(offline["OMP_NUM_THREADS"] == "1", "OMP_NUM_THREADS must be 1")
    _require(offline["MKL_NUM_THREADS"] == "1", "MKL_NUM_THREADS must be 1")

    gates = data["acceptance_gates"]
    for flag in FORBIDDEN_ACCEPTANCE_GATES:
        _require(gates[flag] is False, "acceptance_gates." + flag + " must be false")
    _require(gates["input_count_exact"] is True, "input_count_exact must be true")
    _require(gates["cache_revision_exact"] is True, "cache_revision_exact must be true")
    _require(gates["dense_vector_count"] == EXPECTED_CHILD_CHUNK_COUNT,
             "dense_vector_count must be 12")
    _require(gates["sparse_vector_count"] == EXPECTED_CHILD_CHUNK_COUNT,
             "sparse_vector_count must be 12")

    publishability = data["publishability"]
    _require(publishability["publishable_result"] is False,
             "publishable_result must be false")
    _require(publishability["retrieval_effectiveness_claimed"] is False,
             "retrieval_effectiveness_claimed must be false")

    print("[OK] Pilot embedding execution contract")
    print("  contract id:", contract["id"])
    print("  contract version:", contract["version"])
    print("  status:", contract["status"])
    print("  model:", model["model_id"])
    print("  revision:", revision)
    print("  device:", execution["device"])
    print("  batch sizes:", benchmark["batch_sizes"])
    print("  expected chunks:", source["expected_child_chunk_count"])
    print("  embedding enabled:", execution["embedding_enabled"])
    print("  model download enabled:", execution["model_download_enabled"])
    print("  network access enabled:", execution["network_access_enabled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())