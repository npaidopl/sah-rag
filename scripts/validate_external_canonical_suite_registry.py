"""Validate the Step 92 external canonical suite registry."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/external_canonical_suite_registry.yaml")
DATASETS = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(
        data.get("artifact_type") == "external_canonical_suite_registry",
        "artifact_type changed",
    )

    suite = data["suite"]
    _require(suite["registration_status"] == "REGISTERED", "registration_status must be REGISTERED")
    _require(suite["dataset_count"] == 7, "dataset_count must be 7")
    _require(suite["total_documents"] == 8879, "total_documents must be 8879")
    _require(suite["total_examples"] == 29718, "total_examples must be 29718")
    _require(
        suite["collection_fingerprint"]
        == "5f577978b6ab7a4ee3fef29b68db29f95b37261f103bb6cfd5fd7094c70a2aa4",
        "collection_fingerprint must match",
    )

    ds = data["datasets"]
    _require(set(ds.keys()) == set(DATASETS), "dataset set changed")
    for name in DATASETS:
        d = ds[name]
        _require(d["status"] == "REGISTERED", f"{name}: status must be REGISTERED")
        _require(
            d["document_count_observed"] == d["document_unique_ids"],
            f"{name}: document IDs not unique",
        )
        _require(
            d["query_unique_ids"] == d["example_count_observed"],
            f"{name}: query IDs not unique",
        )
        _require(
            d["unresolved_query_doc_refs"] == 0,
            f"{name}: unresolved query.doc_ids",
        )
        _require(
            d["unresolved_gt_relevant_refs"] == 0,
            f"{name}: unresolved ground_truth.relevant_ids",
        )

    integrity = data["integrity"]
    for flag in (
        "dataset_set_exact",
        "collection_fingerprint_matches",
        "all_manifests_valid",
        "all_validations_pass",
        "all_document_ids_unique",
        "all_query_ids_unique",
        "all_query_document_links_resolve",
        "all_relevant_ids_resolve",
        "all_source_files_unchanged",
    ):
        _require(integrity[flag] is True, "integrity." + flag + " must be true")

    print("[OK] External canonical suite registry")
    print("  repository commit:", suite["repository_commit"])
    print("  datasets:", len(ds))
    print("  total documents:", suite["total_documents"])
    print("  total examples:", suite["total_examples"])
    print("  fingerprint valid:", integrity["collection_fingerprint_matches"])
    print("  manifests valid:", integrity["all_manifests_valid"])
    print("  validations passed:", integrity["all_validations_pass"])
    print("  query links valid:", integrity["all_query_document_links_resolve"])
    print("  relevant links valid:", integrity["all_relevant_ids_resolve"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())