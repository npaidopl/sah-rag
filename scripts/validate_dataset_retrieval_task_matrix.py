"""Validate the Step 93 dataset retrieval task matrix."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/dataset_retrieval_task_matrix.yaml")
DATASETS = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")
SCOPES = {"GLOBAL_DATASET_CORPUS","SHARED_DOMAIN_CORPUS",
          "QUERY_SPECIFIC_CORPUS","QUERY_DOCUMENT_BUNDLE"}
UNITS = {"CANONICAL_DOCUMENT","CANONICAL_CHUNK","SOURCE_SPAN","PASSAGE","NEEDLE_OR_EVIDENCE_ITEM"}


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "dataset_retrieval_task_matrix", "artifact_type changed")

    ds = d["datasets"]
    _require(set(ds.keys()) == set(DATASETS), "dataset set changed")

    for name in DATASETS:
        r = ds[name]
        _require(r["corpus_scope"] in SCOPES, f"{name}: invalid corpus_scope")
        _require(
            r["query_text_projection"]["supplied_context_in_query_embedding"] is False,
            f"{name}: supplied_context_in_query_embedding must be false",
        )
        _require(r["primary_retrieval_unit"] in UNITS, f"{name}: invalid primary_retrieval_unit")
        _require(r["primary_evaluation_unit"] in UNITS, f"{name}: invalid primary_evaluation_unit")
        _require("permitted_corpus_selector" in r, f"{name}: permitted_corpus_selector missing")
        _require("leakage_prohibitions" in r, f"{name}: leakage_prohibitions missing")
        _require(len(r["leakage_prohibitions"]) >= 1, f"{name}: must have leakage prohibitions")

    # CUAD specifics
    cuad_policy = d["cuad_evaluation_policy"]
    _require(
        cuad_policy["positive_evidence_subset"]["query_count"] == 6702,
        "CUAD positive count must be 6702",
    )
    _require(
        cuad_policy["zero_evidence_subset"]["query_count"] == 14208,
        "CUAD zero count must be 14208",
    )
    _require(cuad_policy["combined_reporting"]["separate_subsets_required"] is True,
             "CUAD must separate subsets")

    # RAG system tuning policy
    tuning = d["rag_system_tuning"]
    _require(tuning["test_data_for_tuning_prohibited"] is True, "must prohibit test-data tuning")
    _require(tuning["selected_configuration_frozen_before_test"] is True, "must freeze config")

    print("[OK] Dataset retrieval task matrix")
    print("  datasets:", len(ds))
    print("  CUAD positive/zero:", 6702, "/", 14208)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())