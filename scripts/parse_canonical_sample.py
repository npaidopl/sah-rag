"""Parse a canonical dataset with the deterministic structural parser."""

from __future__ import annotations

import argparse
from pathlib import Path

from sah_rag.artifacts import (
    ArtifactComponentReference,
    ArtifactSidecarManifest,
    sha256_file,
    write_json_model_atomic,
)
from sah_rag.canonical.jsonl import iter_jsonl
from sah_rag.parsing import (
    DeterministicStructuralParser,
    parse_canonical_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Parse a canonical dataset "
            "with the deterministic structural parser."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to parse.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path for parsed documents.",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional artifact-sidecar output path.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs.",
    )

    return parser.parse_args()


def main() -> None:
    """Execute canonical parsing and publish artifacts."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    parser_instance = DeterministicStructuralParser()

    summary = parse_canonical_dataset(
        dataset_id=args.dataset,
        parser=parser_instance,
        output_path=args.output,
        limit=args.limit,
        overwrite=args.overwrite,
    )

    print("[OK] Dataset:", summary.dataset_id)
    print("[OK] Documents:", summary.document_count)
    print("[OK] Parents:", summary.parent_count)
    print("[OK] Empty documents:", summary.empty_document_count)
    print("[OK] Written:", summary.output_path.resolve())

    if args.manifest is not None:
        records = list(iter_jsonl(summary.output_path))
        artifact_sha = sha256_file(summary.output_path)
        file_size = summary.output_path.stat().st_size

        component = ArtifactComponentReference(
            component_type="parser",
            component_id="DeterministicStructuralParser",
            component_version="0.3.0",
            configuration_hash="clerk-full-corpus",
        )

        manifest = ArtifactSidecarManifest(
            artifact_type="parsed_document_corpus",
            artifact_path=str(summary.output_path.name),
            artifact_size_bytes=file_size,
            artifact_sha256=artifact_sha,
            record_count=len(records),
            component=component,
            metadata={
                "dataset": summary.dataset_id,
                "document_limit": args.limit,
            },
        )

        write_json_model_atomic(
            args.manifest,
            manifest,
            overwrite=args.overwrite,
        )
        print("[OK] Manifest:", args.manifest.resolve())


if __name__ == "__main__":
    main()


