"""Inspect canonical datasets before defining strict schemas."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rag_system.canonical.inspection import (
    collect_key_types,
    describe_record_structure,
)
from rag_system.canonical.jsonl import iter_jsonl, read_first_record
from rag_system.canonical.paths import (
    get_canonical_root,
    list_dataset_ids,
    resolve_dataset_paths,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect canonical dataset schemas using streaming JSONL reads."
        )
    )

    parser.add_argument(
        "--sample-limit",
        type=int,
        default=100,
        help="Number of records used for key/type profiling.",
    )

    parser.add_argument(
        "--full-id-scan",
        action="store_true",
        help=(
            "Scan all records for ID uniqueness and reference statistics. "
            "This can take longer on the largest datasets."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/canonical_schema_report.json"),
        help="Output JSON report path.",
    )

    return parser.parse_args()


def extract_query_id(record: dict[str, Any]) -> str | None:
    """Extract query_id from supported canonical shapes."""

    direct_id = record.get("query_id")

    if isinstance(direct_id, str):
        return direct_id

    query = record.get("query")

    if isinstance(query, dict):
        nested_id = query.get("query_id")

        if isinstance(nested_id, str):
            return nested_id

    return None


def extract_query_doc_ids(record: dict[str, Any]) -> list[str]:
    """ry.doc_ids when present."""

    query = record.get("query")

    if not isinstance(query, dict):
        return []

    doc_ids = query.get("doc_ids")

    if not isinstance(doc_ids, list):
        return []

    return [
        value
        for value in doc_ids
        if isinstance(value, str)
    ]


def scan_document_ids(
    path: Path,
) -> tuple[int, int, set[str], Counter[str]]:
    """Scan canonical documents and return ID statistics."""

    record_count = 0
    missing_id_count = 0
    doc_ids: set[str] = set()
    duplicate_ids: Counter[str] = Counter()

    for record in iter_jsonl(path):
        record_count += 1
        doc_id = record.get("doc_id")

        if not isinstance(doc_id, str) or not doc_id:
            missing_id_count += 1
            continue

        if doc_id in doc_ids:
            duplicate_ids[doc_id] += 1
        else:
            doc_ids.add(doc_id)

    return (
        record_count,
        missing_id_count,
        doc_ids,
        duplicate_ids,
    )


def scan_example_ids_and_references(
    path: Path,
    known_doc_ids: set[str],
) -> dict[str, Any]:
    """Scan examples for ID uniqueness and doc reference resolution."""

    record_count = 0
    missing_query_id_count = 0
    query_ids: set[str] = set()
    duplicate_query_ids: Counter[str] = Counter()

    referenced_doc_id_count = 0
    unresolved_doc_ids: Counter[str] = Counter()
    ground_truth_types: Counter[str] = Counter()

    for record in iter_jsonl(path):
        record_count += 1

        query_id = extract_query_id(record)

        if query_id is None:
            missing_query_id_count += 1
        elif query_id in query_ids:
            duplicate_query_ids[query_id] += 1
        else:
            query_ids.add(query_id)

        for doc_id in extract_query_doc_ids(record):
            referenced_doc_id_count += 1

            if doc_id not in known_doc_ids:
                unresolved_doc_ids[doc_id] += 1

        ground_truth = record.get("ground_truth")

        if isinstance(ground_truth, dict):
            ground_truth_type = ground_truth.get(
                "ground_truth_type"
            )

            if isinstance(ground_truth_type, str):
                ground_truth_types[ground_truth_type] += 1

    return {
        "record_count": record_count,
        "missing_query_id_count": missing_query_id_count,
        "unique_query_id_count": len(query_ids),
        "duplicate_query_id_count": sum(
            duplicate_query_ids.values()
        ),
        "duplicate_query_id_examples": dict(
            duplicate_query_ids.most_common(10)
        ),
        "referenced_doc_id_count": referenced_doc_id_count,
        "unresolved_doc_id_count": sum(
            unresolved_doc_ids.values()
        ),
        "unresolved_doc_id_examples": dict(
            unresolved_doc_ids.most_common(10)
        ),
        "ground_truth_types": dict(
            sorted(ground_truth_types.items())
        ),
    }


def build_dataset_report(
    dataset_id: str,
    *,
    sample_limit: int,
    full_id_scan: bool,
) -> dict[str, Any]:
    """Build a report for one dataset."""

    paths = resolve_dataset_paths(dataset_id)
    missing_files = paths.missing_files()

    report: dict[str, Any] = {
        "dataset_id": dataset_id,
        "root": str(paths.root),
        "complete": not missing_files,
        "missing_files": [
            str(path)
            for path in missing_files
        ],
        "file_sizes_bytes": {
            "documents": (
                paths.documents.stat().st_size
                if paths.documents.is_file()
                else None
            ),
            "examples": (
                paths.examples.stat().st_size
                if paths.examples.is_file()
                else None
            ),
            "validation": (
                paths.validation.stat().st_size
                if paths.validation.is_file()
                else None
            ),
            "manifest": (
                paths.manifest.stat().st_size
                if paths.manifest.is_file()
                else None
            ),
        },
    }

    if missing_files:
        return report

    first_document = read_first_record(paths.documents)
    first_example = read_first_record(paths.examples)

    report["document_top_level_types"] = collect_key_types(
        iter_jsonl(paths.documents),
        sample_limit=sample_limit,
    )

    report["example_top_level_types"] = collect_key_types(
        iter_jsonl(paths.examples),
        sample_limit=sample_limit,
    )

    report["first_document_structure"] = (
        describe_record_structure(first_document)
    )

    report["first_example_structure"] = (
        describe_record_structure(first_example)
    )

    with paths.validation.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        validation_data = json.load(handle)

    with paths.manifest.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        manifest_data = json.load(handle)

    report["validation_structure"] = describe_record_structure(
        validation_data
        if isinstance(validation_data, dict)
        else {"value": validation_data}
    )

    report["manifest_structure"] = describe_record_structure(
        manifest_data
        if isinstance(manifest_data, dict)
        else {"value": manifest_data}
    )

    if full_id_scan:
        (
            document_count,
            missing_doc_id_count,
            doc_ids,
            duplicate_doc_ids,
        ) = scan_document_ids(paths.documents)

        report["document_scan"] = {
            "record_count": document_count,
            "missing_doc_id_count": missing_doc_id_count,
            "unique_doc_id_count": len(doc_ids),
            "duplicate_doc_id_count": sum(
                duplicate_doc_ids.values()
            ),
            "duplicate_doc_id_examples": dict(
                duplicate_doc_ids.most_common(10)
            ),
        }

        report["example_scan"] = (
            scan_example_ids_and_references(
                paths.examples,
                known_doc_ids=doc_ids,
            )
        )

    return report


def main() -> int:
    """Run canonical schema inspection."""

    args = parse_args()

    if args.sample_limit < 1:
        raise ValueError("--sample-limit must be at least 1.")

    canonical_root = get_canonical_root()
    dataset_ids = list_dataset_ids(canonical_root)

    print(f"[INFO] Canonical root: {canonical_root}")
    print(f"[INFO] Datasets found: {len(dataset_ids)}")

    report: dict[str, Any] = {
        "canonical_root": str(canonical_root),
        "dataset_count": len(dataset_ids),
        "dataset_ids": list(dataset_ids),
        "sample_limit": args.sample_limit,
        "full_id_scan": args.full_id_scan,
        "datasets": {},
    }

    for dataset_id in dataset_ids:
        print(f"[INSPECTING] {dataset_id}")

        dataset_report = build_dataset_report(
            dataset_id,
            sample_limit=args.sample_limit,
            full_id_scan=args.full_id_scan,
        )

        report["datasets"][dataset_id] = dataset_report

        if dataset_report["complete"]:
            print(f"[OK] {dataset_id}")
        else:
            print(f"[INCOMPLETE] {dataset_id}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    print(f"[OK] Written: {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())