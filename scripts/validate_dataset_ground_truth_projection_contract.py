"""Validate the Step 92 dataset ground-truth projection contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/dataset_ground_truth_projection_contract.yaml")
DATASETS = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")
ALLOWED_UNITS = {
    "CANONICAL_DOCUMENT",
    "CANONICAL_CHUNK",
    "SOURCE_SPAN",
    "PASSAGE",
    "NEEDLE_OR_EVIDENCE_ITEM",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(
        data.get("artifact_type") == "dataset_ground_truth_projection_contract",
        "artifact_type changed",
    )
    _require(
        set(data["allowed_evaluation_units"]) == ALLOWED_UNITS,
        "allowed_evaluation_units changed",
    )

    ds = data["datasets"]
    _require(set(ds.keys()) == set(DATASETS), "dataset set changed")
    for name in DATASETS:
        d = ds[name]
        _require(
            d["primary_evaluation_unit"] in ALLOWED_UNITS,
            f"{name}: invalid primary_evaluation_unit",
        )
        _require(
            d["answer_fields_used_for_retrieval"] is False,
            f"{name}: answer_fields_used_for_retrieval must be false",
        )
        _require(
            d["ground_truth_used_during_indexing"] is False,
            f"{name}: ground_truth_used_during_indexing must be false",
        )
        _require(
            d["ground_truth_used_during_querying"] is False,
            f"{name}: ground_truth_used_during_querying must be false",
        )

    rules = data["global_rules"]
    for flag in (
        "ground_truth_available_to_retrieval",
        "ground_truth_available_to_indexing",
        "ground_truth_available_to_querying",
    ):
        _require(rules[flag] is False, "global_rules." + flag + " must be false")
    _require(
        rules["ground_truth_available_only_to_evaluator"] is True,
        "ground_truth_available_only_to_evaluator must be true",
    )
    _require(
        rules["duplicate_document_identities_remain_distinct"] is True,
        "duplicate_document_identities_remain_distinct must be true",
    )

    # CUAD zero-relevance policy
    cuad = ds["cuad"]
    _require(
        cuad.get("zero_relevance_queries_supported") is True,
        "CUAD must support zero-relevance queries",
    )

    print("[OK] Dataset ground-truth projection contract")
    print("  datasets:", len(ds))
    from collections import Counter
    units = Counter(d["primary_evaluation_unit"] for d in ds.values())
    print("  primary unit distribution:", dict(units))
    print("  ground truth available to retrieval: False")
    print("  ground truth available only to evaluator: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())