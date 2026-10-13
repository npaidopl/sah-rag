"""Validate the Step 88 pilot judgment mapping contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_judgment_mapping_contract.yaml")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "mapping contract must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_judgment_mapping_contract",
        "artifact_type changed",
    )
    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_document_judgments",
        "contract.id changed",
    )

    source = data["source"]
    _require(source["expected_query_count"] == 6, "expected_query_count must be 6")
    _require(source["expected_judgment_count"] == 12, "expected_judgment_count must be 12")
    _require(
        source["expected_judgments_per_query"] == 2,
        "expected_judgments_per_query must be 2",
    )

    identity = data["identity"]
    _require(identity["query_field"] == "query_id", "query_field must be query_id")
    _require(identity["relevant_item_field"] == "item_id", "relevant_item_field must be item_id")
    _require(identity["relevance_field"] == "relevance", "relevance_field must be relevance")
    _require(
        data["evaluation_unit"] == "canonical_document",
        "evaluation_unit must be canonical_document",
    )
    _require(
        data["retrieved_identity_field"] == "doc_id",
        "retrieved_identity_field must be doc_id",
    )

    rel = data["relevance"]
    _require(rel["type"] == "binary", "relevance.type must be binary")
    _require(rel["relevant_value"] == 1, "relevant_value must be 1")

    c = data["constraints"]
    _require(c["every_query_has_judgments"] is True, "every_query_has_judgments must be true")
    _require(
        c["duplicate_content_documents_remain_distinct"] is True,
        "duplicate_content_documents_remain_distinct must be true",
    )
    _require(
        c["content_based_judgment_expansion_prohibited"] is True,
        "content_based_judgment_expansion_prohibited must be true",
    )

    leak = data["leakage"]
    for flag in (
        "judgments_available_to_retrieval",
        "judgments_available_to_query_embedding",
        "judgments_available_to_parent_resolution",
        "judgments_available_to_document_resolution",
    ):
        _require(leak[flag] is False, "leakage." + flag + " must be false")
    _require(
        leak["judgments_available_only_to_evaluator"] is True,
        "judgments_available_only_to_evaluator must be true",
    )

    print("[OK] Pilot judgment mapping contract")
    print("  queries:", source["expected_query_count"])
    print("  judgments:", source["expected_judgment_count"])
    print("  judgments per query:", source["expected_judgments_per_query"])
    print("  canonical-document identity: True")
    print("  duplicate documents distinct: True")
    print("  judgments available only to evaluator: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())