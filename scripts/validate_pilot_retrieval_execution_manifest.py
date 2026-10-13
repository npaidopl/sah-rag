"""Validate the Step 87 retrieval execution manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_retrieval_execution_manifest.yaml")
EXPECTED_ID = "pilot-bounded-hybrid-retrieval-v1"
ALLOWED_STATUSES = ("PLANNED", "SUCCEEDED", "INTERRUPTED")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "manifest must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_retrieval_execution_manifest",
        "artifact_type changed",
    )

    execution = data["execution"]
    _require(execution["execution_id"] == EXPECTED_ID, "execution_id changed")
    status = execution["status"]
    _require(status in ALLOWED_STATUSES, "status must be valid")

    approval = data["approval"]
    _require(approval["query_embedding_approved"] is True, "query_embedding_approved must be true")
    _require(
        approval["hybrid_retrieval_approved"] is True,
        "hybrid_retrieval_approved must be true",
    )
    _require(approval["evaluation_approved"] is False, "evaluation_approved must be false")
    _require(approval["generation_approved"] is False, "generation_approved must be false")

    plan = data["plan"]
    _require(plan["query_count"] == 6, "query_count must be 6")
    _require(
        plan["collection_name"]
        == "rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0",
        "collection_name changed",
    )

    results = data["results"]
    if status == "PLANNED":
        _require(results["model_loaded"] is False, "model_loaded must be false")
        _require(
            results["query_embeddings_created"] is False,
            "query_embeddings_created must be false",
        )
        _require(results["evaluation_executed"] is False, "evaluation_executed must be false")
        _require(results["generation_executed"] is False, "generation_executed must be false")
    elif status == "SUCCEEDED":
        _require(results["model_loaded"] is True, "model_loaded must be true")
        _require(
            results["query_embeddings_created"] is True,
            "query_embeddings_created must be true",
        )
        _require(results["query_embedding_count"] == 6, "query_embedding_count must be 6")
        _require(results["round_trip_valid"] is True, "round_trip_valid must be true")
        _require(results["checksums_valid"] is True, "checksums_valid must be true")
        _require(results["judgments_loaded"] is False, "judgments_loaded must be false")
        _require(results["evaluation_executed"] is False, "evaluation_executed must be false")
        _require(results["generation_executed"] is False, "generation_executed must be false")

    print("[OK] Pilot retrieval execution manifest")
    print("  execution id:", execution["execution_id"])
    print("  status:", status)
    print("  query count:", plan["query_count"])
    print("  collection:", plan["collection_name"])
    print("  evaluation approved:", approval["evaluation_approved"])
    print("  generation approved:", approval["generation_approved"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())