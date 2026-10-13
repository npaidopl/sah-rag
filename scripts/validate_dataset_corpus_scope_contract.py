"""Validate the Step 92 dataset corpus scope contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/dataset_corpus_scope_contract.yaml")
DATASETS = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")
ALLOWED_SCOPES = {
    "GLOBAL_DATASET_CORPUS",
    "SHARED_DOMAIN_CORPUS",
    "QUERY_SPECIFIC_CORPUS",
    "QUERY_DOCUMENT_BUNDLE",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(
        data.get("artifact_type") == "dataset_corpus_scope_contract",
        "artifact_type changed",
    )
    _require(
        set(data["allowed_scopes"]) == ALLOWED_SCOPES,
        "allowed_scopes changed",
    )

    ds = data["datasets"]
    _require(set(ds.keys()) == set(DATASETS), "dataset set changed")
    for name in DATASETS:
        d = ds[name]
        _require(
            d["expected_scope"] in ALLOWED_SCOPES,
            f"{name}: invalid expected_scope",
        )
        _require(
            "corpus_selector" in d,
            f"{name}: corpus_selector missing",
        )

    risks = data["trivial_document_selection_risks"]
    for name in DATASETS:
        _require(name in risks, f"trivial_document_selection_risks missing {name}")

    print("[OK] Dataset corpus-scope contract")
    print("  datasets:", len(ds))
    from collections import Counter
    counts = Counter(d["expected_scope"] for d in ds.values())
    print("  scope distribution:", dict(counts))
    print("  trivial document-selection risks recorded:", len(risks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())