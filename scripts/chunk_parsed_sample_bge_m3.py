"""Create BGE-M3-tokenized chunks from parsed artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from sah_rag.chunking import (
    ChildChunkerConfig,
    HierarchicalChildChunker,
    HuggingFaceOffsetTokenizer,
    chunk_parsed_artifacts,
)

MODEL_ID = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create BGE-M3-compatible child chunks "
            "from parsed-document JSONL."
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
        help=(
            "Total model-input token budget, including "
            "special tokens."
        ),
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=32,
        help="Overlap measured in content tokens.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional output sidecar path.",
    )

    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=None,
        help="Optional parsed-artifact sidecar.",
    )


    return parser.parse_args()


def main() -> None:
    """Chunk parsed artifacts with the cached BGE-M3 tokenizer."""

    args = parse_args()

    load_dotenv(
        dotenv_path=Path(".env"),
        override=False,
    )

    hf_home = os.getenv("HF_HOME")

    if not hf_home:
        raise RuntimeError(
            "HF_HOME is not configured in .env."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=hf_home,
        use_fast=True,
        trust_remote_code=False,
        local_files_only=True,
    )

    offset_tokenizer = HuggingFaceOffsetTokenizer(
        tokenizer,
        model_id=MODEL_ID,
    )

    chunker = HierarchicalChildChunker(
        tokenizer=offset_tokenizer,
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
        manifest_path=args.manifest,
        input_manifest_path=args.input_manifest,
        execution_profile="local_cpu",
        manifest_metadata={
            "sample": True,
            "model_id": MODEL_ID,
        },
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
        "[OK] Model-input budget:",
        chunker.config.chunk_size,
    )
    print(
        "[OK] Content capacity:",
        chunker.content_chunk_size,
    )
    print(
        "[OK] Special tokens:",
        offset_tokenizer.special_tokens_count,
    )
    print(
        "[OK] Tokenizer:",
        offset_tokenizer.tokenizer_id,
    )
    print(
        "[OK] Chunker version:",
        chunker.config.chunker_version,
    )
    print(
        "[OK] Config hash:",
        chunker.config_hash,
    )
    print(
        "[OK] Written:",
        summary.output_path.resolve(),
    )

    if summary.manifest_path is not None:
        print(
            "[OK] Manifest:",
            summary.manifest_path.resolve(),
        )


if __name__ == "__main__":
    main()
