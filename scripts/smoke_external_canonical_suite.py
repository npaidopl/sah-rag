"""Fast smoke check for the Step 92 registration."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("[FAIL] " + message)


def main() -> int:
    registry = yaml.safe_load(
        (REPO_ROOT / "baseline_audit" / "external_canonical_suite_registry.yaml")
        .read_text(encoding="utf-8")
    )
    manifest = yaml.safe_load(
        (REPO_ROOT / "baseline_audit" / "external_canonical_suite_manifest.yaml")
        .read_text(encoding="utf-8")
    )

    _require(manifest["execution"]["status"] == "SUCCEEDED", "manifest status must be SUCCEEDED")
    _require(registry["suite"]["dataset_count"] == 7, "dataset_count must be 7")
    _require(registry["suite"]["total_documents"] == 8879, "total_documents must be 8879")
    _require(registry["suite"]["total_examples"] == 29718, "total_examples must be 29718")

    integrity = registry["integrity"]
    for flag in (
        "dataset_set_exact",
        "collection_fingerprint_matches",
        "all_manifests_valid",
        "all_validations_pass",
        "all_document_ids_unique",
        "all_query_ids_unique",
        "all_query_document_links_resolve",
        "all_relevant_ids_resolve",
    ):
        _require(integrity[flag] is True, "integrity." + flag + " must be true")

    print("[OK] External canonical suite smoke")
    print("  repository commit:", registry["suite"]["repository_commit"])
    print("  datasets:", registry["suite"]["dataset_count"])
    print("  documents:", registry["suite"]["total_documents"])
    print("  examples:", registry["suite"]["total_examples"])
    print("  fingerprint:", integrity["collection_fingerprint_matches"])
    print("  adapter executed: False")
    print("  parser executed: False")
    print("  tokenizer loaded: False")
    print("  model loaded: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())