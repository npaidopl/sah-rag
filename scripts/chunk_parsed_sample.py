"""Chunk a parsed-document JSONL sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_system.chunking import (
    ChildChunkerConfig,
    HierarchicalChildChunker,
    RegexOffsetTokenizer,
    chunk_parsed_artifacts,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert parsed-document JSONL into child chunks."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Run sample child chunking."""

    args = parse_args()

    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        ),
    )

    summary = chunk_parsed_artifacts(
        input_path=args.input,
        output_path=args.output,
        chunker=chunker,
        overwrite=args.overwrite,
    )

    print(
        "[OK] Parsed documents:",
        summary.parsed_document_count,
    )
    print("[OK] Parents:", summary.parent_count)
    print("[OK] Chunks:", summary.chunk_count)
    print(
        "[OK] Parents without chunks:",
        summary.parents_without_chunks,
    )
    print(
        "[OK] Chunker version:",
        chunker.config.chunker_version,
    )
    print(
        "[OK] Tokenizer:",
        chunker.tokenizer.tokenizer_id,
    )
    print(
        "[OK] Chunker config hash:",
        chunker.config_hash,
    )
    print(
        "[OK] Written:",
        summary.output_path.resolve(),
    )


if __name__ == "__main__":
    main()
