"""Diagnose deterministic parser boundary behavior."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sah_rag.canonical import (
    iter_canonical_documents,
    list_dataset_ids,
)
from sah_rag.parsing import (
    DeterministicStructuralParser,
    detect_heading_reason,
    iter_paragraph_spans,
)

DEFAULT_OUTPUT = Path(
    "outputs/parser_boundary_diagnostics.json"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect heading rules, smallest parents, largest "
            "parents, and documents with extreme parent counts."
        )
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def safe_preview(
    text: str,
    *,
    maximum_characters: int = 180,
) -> str:
    """Create a single-line diagnostic preview."""
    compact = " ".join(text.split())
    return compact[:maximum_characters]


def main() -> None:
    """Create parser diagnostics across canonical datasets."""
    args = parse_args()
    if args.limit_per_dataset < 1:
        raise ValueError(
            "--limit-per-dataset must be at least 1."
        )
    if args.examples_per_category < 1:
        raise ValueError(
            "--examples-per-category must be at least 1."
        )

    parser = DeterministicStructuralParser()
    report: dict[str, Any] = {
        "parser_config_hash": parser.config_hash,
        "limit_per_dataset": args.limit_per_dataset,
        "examples_per_category": (
            args.examples_per_category
        ),
        "datasets": {},
    }

    for dataset_id in list_dataset_ids():
        print(f"[DIAGNOSING] {dataset_id}")
        heading_reason_counts: Counter[str] = Counter()
        heading_examples: dict[str, list[dict[str, Any]]] = {
            "numbered": [],
            "named": [],
            "uppercase": [],
        }
        parent_records: list[dict[str, Any]] = []
        document_records: list[dict[str, Any]] = []

        for index, document in enumerate(
            iter_canonical_documents(dataset_id)
        ):
            if index >= args.limit_per_dataset:
                break

            blocks = iter_paragraph_spans(
                document.text_normalized
            )
            for block in blocks:
                reason = detect_heading_reason(
                    block,
                    parser.config,
                )
                if reason is None:
                    continue
                reason_value = reason.value
                heading_reason_counts[reason_value] += 1
                examples = heading_examples[reason_value]
                if (
                    len(examples)
                    < args.examples_per_category
                ):
                    examples.append(
                        {
                            "doc_id": document.doc_id,
                            "block_index": block.block_index,
                            "start": block.start,
                            "end": block.end,
                            "characters": len(block.text),
                            "words": len(block.text.split()),
                            "preview": safe_preview(block.text),
                        }
                    )

            parsed = parser.parse(document)
            document_records.append(
                {
                    "doc_id": document.doc_id,
                    "document_characters": len(
                        document.text_normalized
                    ),
                    "parent_count": len(parsed.parents),
                }
            )
            for parent in parsed.parents:
                parent_records.append(
                    {
                        "doc_id": document.doc_id,
                        "parent_id": parent.parent_id,
                        "sibling_index": (
                            parent.sibling_index
                        ),
                        "characters": (
                            parent
                            .normalized_char_span
                            .length
                        ),
                        "title": parent.title,
                        "heading_detected": (
                            parent.metadata.get(
                                "heading_detected"
                            )
                            is True
                        ),
                        "preview": safe_preview(
                            parent.text_normalized
                        ),
                    }
                )

        smallest_parents = sorted(
            parent_records,
            key=lambda item: (
                item["characters"],
                item["doc_id"],
                item["sibling_index"],
            ),
        )[: args.examples_per_category]

        largest_parents = sorted(
            parent_records,
            key=lambda item: (
                -item["characters"],
                item["doc_id"],
                item["sibling_index"],
            ),
        )[: args.examples_per_category]

        highest_parent_counts = sorted(
            document_records,
            key=lambda item: (
                -item["parent_count"],
                item["doc_id"],
            ),
        )[: args.examples_per_category]

        report["datasets"][dataset_id] = {
            "heading_reason_counts": dict(
                sorted(heading_reason_counts.items())
            ),
            "heading_examples": heading_examples,
            "smallest_parents": smallest_parents,
            "largest_parents": largest_parents,
            "highest_parent_count_documents": (
                highest_parent_counts
            ),
        }
        print(
            f"[OK] {dataset_id}: "
            f"{len(document_records)} documents, "
            f"{len(parent_records)} parents, "
            f"heading reasons="
            f"{dict(heading_reason_counts)}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[OK] Written: {args.output.resolve()}")


if __name__ == "__main__":
    main()