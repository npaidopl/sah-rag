"""Validate the Qdrant execution manifest (PLANNED or SUCCEEDED)."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_qdrant_execution_manifest.yaml")

EXPECTED_ID = "pilot-qdrant-bounded-ingestion-v1"
EXPECTED_PHYSICAL_NAME = "rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0"
ALLOWED_STATUSES = ("PLANNED", "SUCCEEDED", "INTERRUPTED")

FORBIDDEN_APPROVALS = (
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
        data.get("artifact_type") == "pilot_qdrant_execution_manifest",
        "artifact_type changed",
    )

    execution = data["execution"]
    _require(execution["execution_id"] == EXPECTED_ID, "execution_id changed")
    _require(execution["contract_version"] == "0.1.0", "contract_version must be 0.1.0")
    status = execution["status"]
    _require(
        status in ALLOWED_STATUSES,
        "status must be one of " + ", ".join(ALLOWED_STATUSES),
    )

    approval = data["approval"]
    _require(
        approval["collection_creation_approved"] is True,
        "collection_creation_approved must be true",
    )
    _require(
        approval["vector_ingestion_approved"] is True,
        "vector_ingestion_approved must be true",
    )
    for flag in FORBIDDEN_APPROVALS:
        _require(approval[flag] is False, "approval." + flag + " must be false")

    plan = data["plan"]
    _require(
        plan["physical_collection_name"] == EXPECTED_PHYSICAL_NAME,
        "plan.physical_collection_name changed",
    )
    _require(plan["expected_point_count"] == 12, "plan.expected_point_count must be 12")

    results = data["results"]
    if status == "PLANNED":
        _require(results["client_instantiated"] is False, "client_instantiated must be false")
        _require(results["storage_root_created"] is False, "storage_root_created must be false")
        _require(results["collection_created"] is False, "collection_created must be false")
        _require(results["vectors_ingested"] is False, "vectors_ingested must be false")
        _require(results["point_count"] is None, "point_count must be null")
        _require(results["round_trip_valid"] is False, "round_trip_valid must be false")
        _require(results["checksums_valid"] is False, "checksums_valid must be false")
    elif status == "SUCCEEDED":
        _require(results["client_instantiated"] is True, "client_instantiated must be true")
        _require(results["collection_created"] is True, "collection_created must be true")
        _require(results["vectors_ingested"] is True, "vectors_ingested must be true")
        _require(results["point_count"] == 12, "point_count must be 12")
        _require(
            results["unique_point_id_count"] == 12,
            "unique_point_id_count must be 12",
        )
        _require(
            results["collection_restart_valid"] is True,
            "collection_restart_valid must be true",
        )
        _require(
            results["point_ids_match_plan"] is True,
            "point_ids_match_plan must be true",
        )
        _require(results["payloads_valid"] is True, "payloads_valid must be true")
        _require(results["round_trip_valid"] is True, "round_trip_valid must be true")
        _require(results["checksums_valid"] is True, "checksums_valid must be true")
        _require(isinstance(results["storage_bytes"], int), "storage_bytes must be int")
        _require(
            isinstance(results["wall_clock_seconds"], (int, float)),
            "wall_clock_seconds must be numeric",
        )

    for flag in ("retrieval_executed", "evaluation_executed", "generation_executed"):
        _require(results[flag] is False, "results." + flag + " must be false")

    next_gate = data["next_gate"]
    _require(
        next_gate["required_step"] == "bounded_retrieval_plan",
        "next_gate.required_step must be bounded_retrieval_plan",
    )
    _require(next_gate["approved"] is False, "next_gate.approved must be false")

    print("[OK] Pilot Qdrant execution manifest")
    print("  execution id:", execution["execution_id"])
    print("  status:", status)
    print("  collection creation approved:", approval["collection_creation_approved"])
    print("  ingestion approved:", approval["vector_ingestion_approved"])
    print("  retrieval approved:", approval["retrieval_execution_approved"])
    if status == "SUCCEEDED":
        print("  client instantiated:", results["client_instantiated"])
        print("  collection created:", results["collection_created"])
        print("  vectors ingested:", results["vectors_ingested"])
        print("  point count:", results["point_count"])
        print("  client restart valid:", results["collection_restart_valid"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())