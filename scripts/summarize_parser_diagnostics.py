"""Summarize deterministic parser boundary diagnostics."""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

import json
from pathlib import Path
from typing import Any

REPORT_PATH = Path(
    "outputs/parser_boundary_diagnostics.json"
)


def print_parent(
    label: str,
    parent: dict[str, Any],
) -> None:
    """Print one parent diagnostic entry."""
    print(
        f"{label}: "
        f"doc={parent['doc_id']} "
        f"index={parent['sibling_index']} "
        f"chars={parent['characters']} "
        f"heading={parent['heading_detected']} "
        f"title={parent['title']!r} "
        f"preview={parent['preview']!r}"
    )


def main() -> None:
    """Print concise diagnostics."""
    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )
    print("PARSER BOUNDARY DIAGNOSTICS")
    print(
        "Parser config hash:",
        report["parser_config_hash"],
    )
    print(
        "Limit per dataset:",
        report["limit_per_dataset"],
    )
    print()

    for dataset_id, diagnostics in (
        report["datasets"].items()
    ):
        print("=" * 88)
        print("DATASET:", dataset_id)
        print(
            "Heading reason counts:",
            diagnostics["heading_reason_counts"],
        )
        for reason in (
            "numbered",
            "named",
            "uppercase",
        ):
            examples = diagnostics[
                "heading_examples"
            ][reason]
            print(
                f"{reason.upper()} EXAMPLES:",
                len(examples),
            )
            for example in examples[:5]:
                print(
                    "  ",
                    f"doc={example['doc_id']} "
                    f"block={example['block_index']} "
                    f"chars={example['characters']} "
                    f"words={example['words']} "
                    f"preview={example['preview']!r}",
                )
        print("SMALLEST PARENTS:")
        for parent in diagnostics[
            "smallest_parents"
        ][:5]:
            print_parent("  ", parent)
        print("LARGEST PARENTS:")
        for parent in diagnostics[
            "largest_parents"
        ][:5]:
            print_parent("  ", parent)
        print("HIGHEST PARENT COUNTS:")
        for document in diagnostics[
            "highest_parent_count_documents"
        ][:5]:
            print(
                "  ",
                f"doc={document['doc_id']} "
                f"parents={document['parent_count']} "
                f"characters="
                f"{document['document_characters']}",
            )
        print()


if __name__ == "__main__":
    main()