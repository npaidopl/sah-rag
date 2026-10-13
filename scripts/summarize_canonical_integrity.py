"""Summarize the full canonical integrity report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPORT_PATH = Path(
    "outputs/canonical_full_integrity_report.json"
)


def format_mapping(mapping: dict[str, Any]) -> str:
    """Format a small mapping for console output."""

    if not mapping:
        return "none"

    return ", ".join(
        f"{key}={value}"
        for key, value in sorted(mapping.items())
    )


def main() -> None:
    """Print a concise integrity summary."""

    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"Integrity report not found: {REPORT_PATH}"
        )

    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    print("CANONICAL FULL INTEGRITY SUMMARY")
    print("Root:", report["canonical_root"])
    print("Dataset count:", report["dataset_count"])
    print(
        "Dataset IDs:",
        ", ".join(report["dataset_ids"]),
    )
    print(
        "Full ID scan:",
        report["full_id_scan"],
    )
    print()

    total_documents = 0
    total_examples = 0
    total_missing_doc_ids = 0
    total_duplicate_doc_ids = 0
    total_missing_query_ids = 0
    total_duplicate_query_ids = 0
    total_unresolved_doc_ids = 0

    for dataset_id, dataset in report["datasets"].items():
        document_scan = dataset.get("document_scan", {})
        example_scan = dataset.get("example_scan", {})

        document_count = document_scan.get(
            "record_count",
            0,
        )
        example_count = example_scan.get(
            "record_count",
            0,
        )

        missing_doc_ids = document_scan.get(
            "missing_doc_id_count",
            0,
        )
        duplicate_doc_ids = document_scan.get(
            "duplicate_doc_id_count",
            0,
        )
        missing_query_ids = example_scan.get(
            "missing_query_id_count",
            0,
        )
        duplicate_query_ids = example_scan.get(
            "duplicate_query_id_count",
            0,
        )
        unresolved_doc_ids = example_scan.get(
            "unresolved_doc_id_count",
            0,
        )

        total_documents += document_count
        total_examples += example_count
        total_missing_doc_ids += missing_doc_ids
        total_duplicate_doc_ids += duplicate_doc_ids
        total_missing_query_ids += missing_query_ids
        total_duplicate_query_ids += duplicate_query_ids
        total_unresolved_doc_ids += unresolved_doc_ids

        print("=" * 76)
        print("DATASET:", dataset_id)
        print("Complete:", dataset["complete"])
        print("Documents:", document_count)
        print(
            "Unique document IDs:",
            document_scan.get("unique_doc_id_count", 0),
        )
        print("Missing document IDs:", missing_doc_ids)
        print("Duplicate document IDs:", duplicate_doc_ids)
        print("Examples:", example_count)
        print(
            "Unique query IDs:",
            example_scan.get("unique_query_id_count", 0),
        )
        print("Missing query IDs:", missing_query_ids)
        print("Duplicate query IDs:", duplicate_query_ids)
        print(
            "Referenced document IDs:",
            example_scan.get(
                "referenced_doc_id_count",
                0,
            ),
        )
        print(
            "Unresolved document references:",
            unresolved_doc_ids,
        )
        print(
            "Ground-truth types:",
            format_mapping(
                example_scan.get(
                    "ground_truth_types",
                    {},
                )
            ),
        )
        print()

    print("=" * 76)
    print("TOTALS")
    print("Documents:", total_documents)
    print("Examples:", total_examples)
    print("Missing document IDs:", total_missing_doc_ids)
    print("Duplicate document IDs:", total_duplicate_doc_ids)
    print("Missing query IDs:", total_missing_query_ids)
    print("Duplicate query IDs:", total_duplicate_query_ids)
    print(
        "Unresolved document references:",
        total_unresolved_doc_ids,
    )

    passed = all(
        value == 0
        for value in (
            total_missing_doc_ids,
            total_duplicate_doc_ids,
            total_missing_query_ids,
            total_duplicate_query_ids,
            total_unresolved_doc_ids,
        )
    )

    print()
    print(
        "OVERALL INTEGRITY STATUS:",
        "PASS" if passed else "REVIEW REQUIRED",
    )


if __name__ == "__main__":
    main()