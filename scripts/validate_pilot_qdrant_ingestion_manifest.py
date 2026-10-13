"""Validate the Qdrant ingestion planning manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_qdrant_ingestion_manifest.yaml")

EXPECTED_ID = "pilot-qdrant-ingestion-v1"
EXPECTED_VERSION = "0.1.0"
ALLOWED_STATUSES = ("PLANNED", "SUCCEEDED", "INTERRUPTED")

FORBIDDEN_APPROVALS = (
    "collection_creation_approved",
    "vector_ingestion_approved",
    "retrieval_execution_approved",
    "evaluation_execution_approved",
    "generation_execution_approved",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "manifest must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_qdrant_ingestion_manifest",
        "artifact_type changed",
    )

    execution = data["execution"]
    _require(execution["execution_id"] == EXPECTED_ID, "execution_id changed")
    _require(execution["contract_version"] == EXPECTED_VERSION, "contract_version must be 0.1.0")
    status = execution["status"]
    _require(status in ALLOWED_STATUSES, "status must be one of " + ", ".join(ALLOWED_STATUSES))

    approval = data["approval"]
    _require(
        approval["environment_inspection_approved"] is True,
        "environment_inspection_approved must be true",
    )
    _require(
        approval["ingestion_planning_approved"] is True,
        "ingestion_planning_approved must be true",
    )
    for flag in FORBIDDEN_APPROVALS:
        _require(approval[flag] is False, "approval." + flag + " must be false")

    plan = data["plan"]
    _require(plan["embedding_record_count"] == 12, "plan.embedding_record_count must be 12")
    _require(plan["expected_point_count"] == 12, "plan.expected_point_count must be 12")
    _require(plan["dense_vector_count"] == 12, "plan.dense_vector_count must be 12")
    _require(plan["sparse_vector_count"] == 12, "plan.sparse_vector_count must be 12")
    _require(
        plan["configuration_hash"] is None or (
            isinstance(plan["configuration_hash"], str)
            and len(plan["configuration_hash"]) == 64
        ),
        "plan.configuration_hash must be null or 64-hex",
    )
    _require(
        plan["physical_collection_name"] is None
        or isinstance(plan["physical_collection_name"], str),
        "plan.physical_collection_name must be null or string",
    )

    results = data["results"]
    _require(results["qdrant_client_opened"] is False, "qdrant_client_opened must be false")
    _require(results["storage_root_created"] is False, "storage_root_created must be false")
    _require(results["collection_created"] is False, "collection_created must be false")
    _require(results["vectors_ingested"] is False, "vectors_ingested must be false")
    _require(results["point_count"] is None, "point_count must be null")
    _require(results["round_trip_valid"] is False, "round_trip_valid must be false")
    _require(results["checksums_valid"] is False, "checksums_valid must be false")

    next_gate = data["next_gate"]
    _require(
        next_gate["required_step"] == "bounded_qdrant_ingestion",
        "next_gate.required_step must be bounded_qdrant_ingestion",
    )
    _require(next_gate["approved"] is False, "next_gate.approved must be false")

    print("[OK] Pilot Qdrant ingestion manifest")
    print("  execution id:", execution["execution_id"])
    print("  status:", status)
    print("  expected points:", plan["expected_point_count"])
    print(
        "  configuration hash present:",
        plan["configuration_hash"] is not None,
    )
    print(
        "  physical collection name present:",
        plan["physical_collection_name"] is not None,
    )
    print("  collection creation approved:", approval["collection_creation_approved"])
    print("  ingestion approved:", approval["vector_ingestion_approved"])
    print("  retrieval approved:", approval["retrieval_execution_approved"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())