"""Validate the Step 86 hybrid retrieval contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_retrieval_contract.yaml")

EXPECTED_COLLECTION = "rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0"
FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

FORBIDDEN_EXECUTION_FLAGS = (
    "query_embedding_enabled",
    "dense_retrieval_enabled",
    "sparse_retrieval_enabled",
    "hybrid_retrieval_enabled",
    "parent_aggregation_enabled",
    "document_expansion_enabled",
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
    _require(isinstance(data, dict), "retrieval contract must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_hybrid_retrieval_contract",
        "artifact_type must be pilot_hybrid_retrieval_contract",
    )

    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_bounded_hybrid_retrieval",
        "contract.id changed",
    )
    _require(contract["version"] == "0.1.0", "contract.version must be 0.1.0")
    _require(
        contract["status"] == "FROZEN_FOR_PLANNING",
        "contract.status must be FROZEN_FOR_PLANNING",
    )

    source = data["source"]
    _require(source["expected_query_count"] == 6, "expected_query_count must be 6")
    _require(
        source["query_processing"]["reference_answers_used"] is False,
        "reference_answers_used must be false",
    )
    _require(
        source["query_processing"]["judgments_used"] is False,
        "judgments_used must be false",
    )
    _require(
        source["query_processing"]["relevance_labels_used"] is False,
        "relevance_labels_used must be false",
    )
    _require(
        source["query_processing"]["target_content_group_ids_used_for_filtering"]
        is False,
        "target_content_group_ids must not be used for filtering",
    )
    _require(
        source["query_processing"]["target_content_group_ids_used_for_ranking"]
        is False,
        "target_content_group_ids must not be used for ranking",
    )

    coll = data["collection"]
    _require(coll["backend"] == "Qdrant", "backend must be Qdrant")
    _require(coll["mode"] == "local_persistence", "mode must be local_persistence")
    _require(coll["physical_name"] == EXPECTED_COLLECTION, "physical_name changed")
    _require(coll["expected_point_count"] == 12, "expected_point_count must be 12")

    model = data["model"]
    _require(model["model_id"] == "BAAI/bge-m3", "model_id changed")
    _require(model["model_revision"] == FROZEN_REVISION, "model_revision changed")
    _require(model["batch_size"] == 2, "batch_size must be 2")
    _require(model["local_files_only"] is True, "local_files_only must be true")

    ret = data["retrieval"]
    _require(ret["primary_method"] == "hybrid_rrf", "primary_method must be hybrid_rrf")
    _require(
        ret["diagnostic_methods"] == ["dense", "sparse"],
        "diagnostic_methods must be [dense, sparse]",
    )
    _require(ret["candidate_limit"] == 12, "candidate_limit must be 12")
    _require(ret["child_result_limit"] == 12, "child_result_limit must be 12")
    _require(ret["parent_result_limit"] == 12, "parent_result_limit must be 12")
    _require(ret["document_result_limit"] == 12, "document_result_limit must be 12")
    _require(ret["reranker_enabled"] is False, "reranker_enabled must be false")

    rrf = data["rrf"]
    _require(rrf["implementation"] == "QDRANT_NATIVE_RRF", "rrf implementation changed")
    _require(rrf["rank_constant"] == "UNPINNED", "rrf rank_constant must be UNPINNED")

    cutoff = data["cutoff_policy"]
    _require(cutoff["requested_cutoffs"] == [1, 5, 10, 100], "cutoffs changed")
    _require(
        cutoff["synthetic_padding_prohibited"] is True,
        "synthetic_padding_prohibited must be true",
    )

    execution = data["execution"]
    for flag in FORBIDDEN_EXECUTION_FLAGS:
        _require(execution[flag] is False, "execution." + flag + " must be false")

    pub = data["publishability"]
    _require(pub["publishable_result"] is False, "publishable_result must be false")

    print("[OK] Pilot hybrid retrieval contract")
    print("  id:", contract["id"])
    print("  version:", contract["version"])
    print("  queries:", source["expected_query_count"])
    print("  collection points:", coll["expected_point_count"])
    print("  primary method:", ret["primary_method"])
    print("  candidate limit:", ret["candidate_limit"])
    print("  result limit:", ret["child_result_limit"])
    print("  retrieval enabled:", execution["hybrid_retrieval_enabled"])
    print("  evaluation enabled:", execution["evaluation_enabled"])
    print("  generation enabled:", execution["generation_enabled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())