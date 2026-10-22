"""Validate the frozen Qdrant ingestion contract for the pilot dataset."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_qdrant_ingestion_contract.yaml")

EXPECTED_DATASET = "infrastructure-pilot-synthetic-v1"
EXPECTED_MODEL_ID = "BAAI/bge-m3"
EXPECTED_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

REQUIRED_PROJECT_API = (
    "create_collection",
    "ingest",
    "retrieve_dense",
    "retrieve_sparse",
    "retrieve_hybrid_rrf",
)

FORBIDDEN_EXECUTION_FLAGS = (
    "collection_creation_enabled",
    "vector_ingestion_enabled",
    "retrieval_enabled",
    "evaluation_enabled",
    "generation_enabled",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "ingestion contract must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_qdrant_ingestion_contract",
        "artifact_type must be pilot_qdrant_ingestion_contract",
    )

    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_qdrant_ingestion",
        "contract.id changed",
    )
    _require(contract["version"] == "0.1.0", "contract.version must be 0.1.0")
    _require(
        contract["status"] == "FROZEN_FOR_PLANNING",
        "contract.status must be FROZEN_FOR_PLANNING",
    )

    source = data["source"]
    _require(
        source["expected_embedding_status"] == "SUCCEEDED",
        "expected_embedding_status must be SUCCEEDED",
    )
    _require(source["expected_record_count"] == 12, "expected_record_count must be 12")
    _require(source["expected_shard_count"] == 1, "expected_shard_count must be 1")

    backend = data["backend"]
    _require(backend["type"] == "Qdrant", "backend.type must be Qdrant")
    _require(
        backend["mode"] == "local_persistence",
        "backend.mode must be local_persistence",
    )
    _require(backend["server_required"] is False, "server_required must be false")
    _require(backend["docker_required"] is False, "docker_required must be false")
    _require(backend["network_required"] is False, "network_required must be false")
    _require(
        backend["credentials_required"] is False,
        "credentials_required must be false",
    )

    collection = data["collection"]
    _require(
        collection["logical_name"] == "infrastructure_pilot_rag_system_bge_m3_v1",
        "collection.logical_name changed",
    )
    _require(
        collection["expected_point_count"] == 12,
        "collection.expected_point_count must be 12",
    )

    physical = collection["physical_name"]
    _require(
        physical["function"] == "sah_rag.storage.qdrant.fingerprint_collection_name",
        "physical_name.function changed",
    )
    _require(physical["deterministic"] is True, "physical_name.deterministic must be true")
    _require(
        "spec_divergence" in physical,
        "physical_name must document spec_divergence",
    )

    vectors = data["vectors"]
    _require(vectors["dense"]["name"] == "dense", "dense.name must be dense")
    _require(vectors["dense"]["dimension"] == 1024, "dense.dimension must be 1024")
    _require(vectors["dense"]["distance"] == "COSINE", "dense.distance must be COSINE")
    _require(vectors["sparse"]["name"] == "sparse", "sparse.name must be sparse")
    _require(vectors["sparse"]["indices_sorted"] is True, "sparse.indices_sorted must be true")
    _require(vectors["sparse"]["indices_unique"] is True, "sparse.indices_unique must be true")
    _require(
        vectors["sparse"]["values_non_negative"] is True,
        "sparse.values_non_negative must be true",
    )

    identity = data["point_identity"]
    _require(
        identity["function"] == "sah_rag.storage.qdrant.build_qdrant_point_id",
        "point_identity.function changed",
    )
    _require(identity["algorithm"] == "uuid5", "point_identity.algorithm must be uuid5")
    _require(identity["namespace"] == "URL", "point_identity.namespace must be URL")
    _require(
        identity["input_format"] == "rag-chunk:{chunk_id}",
        "point_identity.input_format changed",
    )

    execution = data["execution"]
    _require(
        execution["environment_inspection_enabled"] is True,
        "environment_inspection_enabled must be true",
    )
    _require(execution["planning_enabled"] is True, "planning_enabled must be true")
    for flag in FORBIDDEN_EXECUTION_FLAGS:
        _require(
            execution[flag] is False,
            "execution." + flag + " must be false",
        )

    api = data["project_api"]
    for name in REQUIRED_PROJECT_API:
        _require(name in api, "project_api missing " + name)

    gates = data["acceptance_gates"]
    for flag in (
        "storage_root_created",
        "collection_creation_performed",
        "ingestion_performed",
        "retrieval_performed",
        "evaluation_performed",
        "generation_performed",
    ):
        _require(gates[flag] is False, "acceptance_gates." + flag + " must be false")

    pub = data["publishability"]
    _require(pub["publishable_result"] is False, "publishable_result must be false")
    _require(
        pub["included_in_final_comparison"] is False,
        "included_in_final_comparison must be false",
    )

    print("[OK] Pilot Qdrant ingestion contract")
    print("  id:", contract["id"])
    print("  version:", contract["version"])
    print("  backend:", backend["type"])
    print("  mode:", backend["mode"])
    print("  expected points:", collection["expected_point_count"])
    print(
        "  dense vector:",
        vectors["dense"]["name"],
        vectors["dense"]["dimension"],
        vectors["dense"]["distance"],
    )
    print("  sparse vector:", vectors["sparse"]["name"])
    print("  collection creation enabled:", execution["collection_creation_enabled"])
    print("  ingestion enabled:", execution["vector_ingestion_enabled"])
    print("  retrieval enabled:", execution["retrieval_enabled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())