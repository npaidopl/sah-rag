"""Print a readable deterministic parser profile."""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

import json
from pathlib import Path

REPORT_PATH = Path(
    "outputs/deterministic_parser_profile.json"
)


def main() -> None:
    """Print the parser profile."""

    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    print("DETERMINISTIC PARSER PROFILE")
    print("Parser:", report["parser_id"])
    print("Version:", report["parser_version"])
    print(
        "Config hash:",
        report["parser_config_hash"],
    )
    print(
        "Limit per dataset:",
        report["limit_per_dataset"],
    )
    print()

    for dataset_id, profile in (
        report["datasets"].items()
    ):
        print("=" * 80)
        print("DATASET:", dataset_id)
        print(
            "Documents:",
            profile["document_count"],
        )
        print(
            "Parents:",
            profile["parent_count"],
        )
        print(
            "Average parents/document:",
            profile[
                "average_parents_per_document"
            ],
        )
        print(
            "Median parents/document:",
            profile[
                "median_parents_per_document"
            ],
        )
        print(
            "Single-parent documents:",
            profile[
                "single_parent_document_count"
            ],
        )
        print(
            "Empty documents:",
            profile[
                "empty_document_count"
            ],
        )
        print(
            "Heading parents:",
            profile[
                "heading_parent_count"
            ],
        )
        print(
            "Very short documents:",
            profile[
                "very_short_document_count"
            ],
        )
        print(
            "Document characters "
            "(min/median/max):",
            profile[
                "minimum_document_characters"
            ],
            profile[
                "median_document_characters"
            ],
            profile[
                "maximum_document_characters"
            ],
        )
        print(
            "Parent characters "
            "(min/median/max):",
            profile[
                "minimum_parent_characters"
            ],
            profile[
                "median_parent_characters"
            ],
            profile[
                "maximum_parent_characters"
            ],
        )
        print()


if __name__ == "__main__":
    main()