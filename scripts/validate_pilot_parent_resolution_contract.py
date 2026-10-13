"""Validate the Step 86 parent/document resolution contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_parent_resolution_contract.yaml")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "parent resolution contract must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_parent_resolution_contract",
        "artifact_type changed",
    )

    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_parent_and_document_resolution",
        "contract.id changed",
    )
    _require(contract["version"] == "0.1.0", "contract.version must be 0.1.0")

    pr = data["parent_resolution"]
    _require(
        pr["implementation"]
        == "rag_system.retrieval.aggregation.aggregate_chunks_to_parents",
        "parent_resolution implementation changed",
    )
    _require(
        pr["aggregation_method"]
        == "MAX_SCORE_PLUS_WEIGHTED_ADDITIONAL_CHILD_SCORE",
        "parent aggregation_method changed",
    )
    _require(
        pr["additional_child_weight"] == 0.05,
        "additional_child_weight must be 0.05",
    )
    _require(
        pr["algorithm_frozen_from_source"] is True,
        "algorithm_frozen_from_source must be true",
    )

    ptb = data["parent_tie_break"]
    _require(ptb["policy"] == "EXPLICIT_DETERMINISTIC", "parent tie_break policy changed")
    _require(
        ptb["key_ordering"]
        == ["parent_score DESC", "best_chunk_rank ASC", "parent_id ASC"],
        "parent tie_break key_ordering changed",
    )
    _require(ptb["frozen_from_source"] is True, "parent tie_break frozen_from_source must be true")

    dr = data["document_resolution"]
    _require(dr["implementation"] == "PROJECT_OWNED_MAPPING", "document implementation changed")
    _require(
        dr["method"] == "MAX_PARENT_SCORE_PER_DOCUMENT",
        "document method must be MAX_PARENT_SCORE_PER_DOCUMENT",
    )
    _require(
        "Duplicate-content documents are not deduplicated"
        in " ".join(dr["requirements"]),
        "document requirements must forbid deduplication",
    )

    dtb = data["document_tie_break"]
    _require(dtb["policy"] == "EXPLICIT_DETERMINISTIC", "document tie_break policy changed")
    _require(
        dtb["key_ordering"]
        == ["document_score DESC", "best_parent_rank ASC", "doc_id ASC"],
        "document tie_break key_ordering changed",
    )
    _require(
        dtb["frozen_by_this_contract"] is True,
        "document tie_break must be frozen_by_this_contract",
    )

    limits = data["limits"]
    _require(limits["parent_result_limit"] == 12, "parent_result_limit must be 12")
    _require(limits["document_result_limit"] == 12, "document_result_limit must be 12")

    print("[OK] Pilot parent resolution contract")
    print("  child source:", data["source"]["retrieval_method"])
    print("  parent aggregation:", pr["aggregation_method"])
    print("  document aggregation:", dr["method"])
    print("  duplicate document deduplication: False")
    print("  parent tie break:", ptb["policy"])
    print("  document tie break:", dtb["policy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())