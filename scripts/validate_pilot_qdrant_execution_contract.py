"""Validate the frozen Step 85 Qdrant execution contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_qdrant_execution_contract.yaml")

FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EXPECTED_PHYSICAL_NAME = "rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "execution contract must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_qdrant_execution_contract",
        "artifact_type must be pilot_qdrant_execution_contract",
    )
    _require(data.get("artifact_version") == "1.0", "artifact_version must be 1.0")

    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_qdrant_bounded_ingestion",
        "contract.id changed",
    )
    _require(contract["version"] == "0.1.0", "contract.version must be 0.1.0")
    _require(
        contract["status"] == "APPROVED_FOR_BOUNDED_EXECUTION",
        "contract.status must be APPROVED_FOR_BOUNDED_EXECUTION",
    )

    backend = data["backend"]
    _require(backend["type"] == "Qdrant", "backend.type must be Qdrant")
    _require(
        backend["mode"] == "local_persistence",
        "backend.mode must be local_persistence",
    )
    _require(
        backend["client_initialization"]["keyword"] == "path",
        "client_initialization.keyword must be path",
    )
    _require(backend["server_required"] is False, "server_required must be false")
    _require(backend["docker_required"] is False, "docker_required must be false")
    _require(backend["network_required"] is False, "network_required must be false")

    source = data["source"]
    _require(
        source["expected_embedding_records"] == 12,
        "expected_embedding_records must be 12",
    )
    _require(source["model_id"] == "BAAI/bge-m3", "model_id must be BAAI/bge-m3")
    _require(
        source["model_revision"] == FROZEN_REVISION,
        "model_revision must equal frozen revision",
    )

    collection = data["collection"]
    _require(
        collection["physical_name"] == EXPECTED_PHYSICAL_NAME,
        "physical_name must be " + EXPECTED_PHYSICAL_NAME,
    )
    _require(
        collection["expected_point_count"] == 12,
        "expected_point_count must be 12",
    )
    _require(
        collection["overwrite_existing_storage"] is False,
        "overwrite_existing_storage must be false",
    )

    vectors = data["vectors"]
    _require(vectors["dense"]["name"] == "dense", "dense.name must be dense")
    _require(vectors["dense"]["dimension"] == 1024, "dense.dimension must be 1024")
    _require(vectors["dense"]["distance"] == "COSINE", "dense.distance must be COSINE")
    _require(vectors["sparse"]["name"] == "sparse", "sparse.name must be sparse")

    execution = data["execution"]
    _require(
        execution["collection_creation_enabled"] is True,
        "collection_creation_enabled must be true",
    )
    _require(
        execution["vector_ingestion_enabled"] is True,
        "vector_ingestion_enabled must be true",
    )
    for flag in (
        "retrieval_enabled",
        "evaluation_enabled",
        "generation_enabled",
        "network_access_enabled",
        "server_mode_enabled",
        "docker_execution_enabled",
    ):
        _require(execution[flag] is False, flag + " must be false")
    _require(
        execution["ingestion_batch_size"] == 12,
        "ingestion_batch_size must be 12",
    )

    limits = data["resource_limits"]
    _require(
        limits["maximum_wall_clock_seconds"] <= 120,
        "maximum_wall_clock_seconds must be <= 120",
    )
    _require(
        limits["maximum_storage_bytes"] <= 104857600,
        "maximum_storage_bytes must be <= 104857600",
    )

    gates = data["acceptance_gates"]
    for flag in (
        "retrieval_execution_performed",
        "evaluation_execution_performed",
        "generation_execution_performed",
    ):
        _require(gates[flag] is False, "acceptance_gates." + flag + " must be false")

    pub = data["publishability"]
    _require(pub["publishable_result"] is False, "publishable_result must be false")

    print("[OK] Pilot Qdrant execution contract")
    print("  id:", contract["id"])
    print("  version:", contract["version"])
    print("  collection creation:", execution["collection_creation_enabled"])
    print("  vector ingestion:", execution["vector_ingestion_enabled"])
    print("  expected points:", collection["expected_point_count"])
    print("  retrieval enabled:", execution["retrieval_enabled"])
    print("  evaluation enabled:", execution["evaluation_enabled"])
    print("  generation enabled:", execution["generation_enabled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())