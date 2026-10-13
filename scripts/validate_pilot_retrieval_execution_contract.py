"""Validate the Step 87 retrieval execution contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_retrieval_execution_contract.yaml")

FORBIDDEN_FLAGS = (
    "reranking_enabled",
    "evaluation_enabled",
    "generation_enabled",
    "network_access_enabled",
    "hosted_requests_enabled",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "contract must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_retrieval_execution_contract",
        "artifact_type changed",
    )
    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_bounded_hybrid_retrieval_execution",
        "contract.id changed",
    )
    _require(contract["version"] == "0.1.0", "contract.version must be 0.1.0")
    _require(
        contract["status"] == "APPROVED_FOR_BOUNDED_EXECUTION",
        "contract.status changed",
    )

    source = data["source"]
    _require(source["expected_query_count"] == 6, "expected_query_count must be 6")
    _require(
        source["judgments_path_access_allowed"] is False,
        "judgments_path_access_allowed must be false",
    )
    _require(
        source["reference_answers_access_allowed"] is False,
        "reference_answers_access_allowed must be false",
    )

    coll = data["collection"]
    _require(
        coll["physical_name"]
        == "rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0",
        "collection name changed",
    )
    _require(coll["expected_point_count"] == 12, "expected_point_count must be 12")

    exec_ = data["execution"]
    for flag in (
        "model_load_enabled",
        "query_embedding_enabled",
        "dense_retrieval_enabled",
        "sparse_retrieval_enabled",
        "hybrid_retrieval_enabled",
        "parent_aggregation_enabled",
        "document_expansion_enabled",
    ):
        _require(exec_[flag] is True, "execution." + flag + " must be true")
    for flag in FORBIDDEN_FLAGS:
        _require(exec_[flag] is False, "execution." + flag + " must be false")

    print("[OK] Pilot retrieval execution contract")
    print("  id:", contract["id"])
    print("  version:", contract["version"])
    print("  query count:", source["expected_query_count"])
    print("  collection:", coll["physical_name"])
    print("  hybrid retrieval enabled:", exec_["hybrid_retrieval_enabled"])
    print("  evaluation enabled:", exec_["evaluation_enabled"])
    print("  generation enabled:", exec_["generation_enabled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())