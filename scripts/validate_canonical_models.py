"""Validate all canonical records against typed models."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rag_system.canonical.loaders import (
    iter_canonical_documents,
    iter_canonical_examples,
)
from rag_system.canonical.paths import list_dataset_ids

OUTPUT_PATH = Path(
    "outputs/canonical_model_validation.json"
)


def main() -> int:
    """Validate every canonical document and example."""

    dataset_ids = list_dataset_ids()

    report: dict[str, Any] = {
        "status": "running",
        "dataset_count": len(dataset_ids),
        "datasets": {},
        "total_documents": 0,
        "total_examples": 0,
    }

    for dataset_id in dataset_ids:
        print(f"[VALIDATING] {dataset_id}")

        document_count = 0
        example_count = 0
        ground_truth_types: Counter[str] = Counter()

        for document in iter_canonical_documents(dataset_id):
            document_count += 1

            if document.dataset != dataset_id:
                raise ValueError(
                    f"{dataset_id}: document {document.doc_id!r} "
                    f"contains dataset={document.dataset!r}"
                )

        for example in iter_canonical_examples(dataset_id):
            example_count += 1

            if example.query.dataset != dataset_id:
                raise ValueError(
                    f"{dataset_id}: query "
                    f"{example.query.query_id!r} contains "
                    f"dataset={example.query.dataset!r}"
                )

            ground_truth_types[
                example.ground_truth.ground_truth_type.value
            ] += 1

        report["datasets"][dataset_id] = {
            "document_count": document_count,
            "example_count": example_count,
            "ground_truth_types": dict(
                sorted(ground_truth_types.items())
            ),
            "status": "pass",
        }

        report["total_documents"] += document_count
        report["total_examples"] += example_count

        print(
            f"[OK] {dataset_id}: "
            f"{document_count} documents, "
            f"{example_count} examples"
        )

    report["status"] = "pass"

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Written: {OUTPUT_PATH.resolve()}")
    print(
        "[TOTAL] "
        f"{report['total_documents']} documents, "
        f"{report['total_examples']} examples"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())