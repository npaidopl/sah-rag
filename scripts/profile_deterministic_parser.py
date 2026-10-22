"""Profile the deterministic parser on canonical samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sah_rag.canonical import (
    iter_canonical_documents,
    list_dataset_ids,
)
from sah_rag.parsing import (
    DeterministicStructuralParser,
    ParserProfileAccumulator,
)

DEFAULT_OUTPUT = Path(
    "outputs/deterministic_parser_profile.json"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Profile deterministic parsing on a limited "
            "sample from every canonical dataset."
        )
    )

    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=20,
        help=(
            "Maximum documents parsed from each dataset."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON report.",
    )

    return parser.parse_args()


def main() -> None:
    """Profile every canonical dataset."""

    args = parse_args()

    if args.limit_per_dataset < 1:
        raise ValueError(
            "--limit-per-dataset must be at least 1."
        )

    parser = DeterministicStructuralParser()

    report: dict[str, Any] = {
        "parser_id": parser.config.parser_id,
        "parser_version": (
            parser.config.parser_version
        ),
        "parser_config_hash": parser.config_hash,
        "limit_per_dataset": (
            args.limit_per_dataset
        ),
        "datasets": {},
    }

    for dataset_id in list_dataset_ids():
        print(f"[PROFILING] {dataset_id}")

        accumulator = ParserProfileAccumulator(
            dataset_id=dataset_id
        )

        for index, document in enumerate(
            iter_canonical_documents(dataset_id)
        ):
            if index >= args.limit_per_dataset:
                break

            parsed_document = parser.parse(document)
            accumulator.add(parsed_document)

        dataset_report = accumulator.to_dict()
        report["datasets"][dataset_id] = dataset_report

        print(
            f"[OK] {dataset_id}: "
            f"{dataset_report['document_count']} documents, "
            f"{dataset_report['parent_count']} parents"
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