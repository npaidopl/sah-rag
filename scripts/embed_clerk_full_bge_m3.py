"""Produce resumable BGE-M3 hybrid embedding shards for the full Clerk chunk artifact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from sah_rag.artifacts import (
    ArtifactSidecarManifest,
    read_json_model,
    sha256_file,
    sha256_fingerprint,
)
from sah_rag.canonical.jsonl import iter_jsonl
from sah_rag.embedding import (
    ChunkEmbeddingRecord,
    EmbeddingKind,
    EmbeddingShardHeader,
    SparseVector,
    VectorPrecision,
    build_embedding_shard_id,
    iter_embedding_records,
    resolve_embedding_shard_paths,
    write_embedding_shard,
)

CHUNKS_PATH = Path("outputs/clerk_chunks_bge_m3_256_full.jsonl")
CHUNKS_MANIFEST_PATH = Path("outputs/clerk_chunks_bge_m3_256_full.manifest.json")
MODEL_REPORT_PATH = Path("outputs/bge_m3_model_download.json")
DEFAULT_OUTPUT_ROOT = Path("outputs/clerk_bge_m3_full_embeddings")

DENSE_DIMENSION = 1024
MAXIMUM_MODEL_INPUT_TOKENS = 256
DATASET_ID = "clerk"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create resumable BGE-M3 hybrid embedding shards "
            "for the complete Clerk chunk artifact."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip complete and validated shards. "
            "Incomplete shards are regenerated."
        ),
    )

    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help=(
            "Optional development limit. "
            "Do not use for the final full run."
        ),
    )

    return parser.parse_args()


def convert_sparse_vector(
    lexical_weights: dict[Any, Any],
) -> SparseVector:
    """Convert BGE lexical weights into sorted sparse arrays."""
    items = sorted(
        (
            int(index),
            float(value),
        )
        for index, value in lexical_weights.items()
        if float(value) != 0.0
    )

    return SparseVector(
        indices=tuple(index for index, _ in items),
        values=tuple(value for _, value in items),
    )


def validate_existing_shard(
    shard_root: Path,
    expected_record_count: int,
) -> bool:
    """Check if an existing shard is complete and valid."""
    shard_paths = resolve_embedding_shard_paths(shard_root)
    if not (
        shard_paths.header.exists()
        and shard_paths.records.exists()
        and shard_paths.manifest.exists()
    ):
        return False

    try:
        manifest = read_json_model(
            shard_paths.manifest, ArtifactSidecarManifest
        )
        if manifest.record_count != expected_record_count:
            return False

        if manifest.artifact_sha256 != sha256_file(shard_paths.records):
            return False

        records = list(iter_embedding_records(shard_paths.records))
        return len(records) == expected_record_count
    except Exception:
        return False


def make_shard_id(
    model_id: str,
    config_hash: str,
    shard_idx: int,
) -> str:
    """Build standardized shard ID."""
    return build_embedding_shard_id(
        dataset=DATASET_ID,
        model_id=model_id,
        embedding_config_hash=config_hash,
        shard_index=shard_idx,
    )


def main() -> None:
    """Generate embedding shards."""
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks artifact missing: {CHUNKS_PATH}")
    if not CHUNKS_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Chunks manifest missing: {CHUNKS_MANIFEST_PATH}")
    if not MODEL_REPORT_PATH.exists():
        raise FileNotFoundError(f"Model report missing: {MODEL_REPORT_PATH}")

    model_report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
    snapshot_path = Path(model_report["snapshot_path"])
    model_id = str(model_report["model_id"])
    model_revision = model_report.get("revision", "5617a9f61b028005a4858fdac845db406aefb181")
    tokenizer_id = f"huggingface:{model_id}@{model_revision}:fast"

    embedding_configuration = {
        "embedder_id": "bge_m3_embedder",
        "embedder_version": "0.1.0",
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": tokenizer_id,
        "embedding_kind": "hybrid",
        "dense_dimension": DENSE_DIMENSION,
        "dense_precision": "float32",
        "dense_normalized": True,
        "maximum_model_input_tokens": MAXIMUM_MODEL_INPUT_TOKENS,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": False,
    }

    config_hash = sha256_fingerprint(embedding_configuration)

    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"Pinned model snapshot missing: {snapshot_path}")

    chunks = list(iter_jsonl(CHUNKS_PATH))
    total_chunks = len(chunks)

    if total_chunks == 0:
        raise ValueError("Chunks artifact is empty.")

    num_shards = math.ceil(total_chunks / args.shard_size)
    if args.limit_shards is not None:
        num_shards = min(num_shards, args.limit_shards)

    shards_to_process: list[tuple[int, list[dict[str, Any]]]] = []
    skipped_shards = 0

    for shard_idx in range(num_shards):
        start_idx = shard_idx * args.shard_size
        end_idx = min(start_idx + args.shard_size, total_chunks)
        shard_chunks = chunks[start_idx:end_idx]
        shard_id = make_shard_id(model_id, config_hash, shard_idx)
        shard_dir = args.output_root / shard_id

        if args.resume and validate_existing_shard(shard_dir, len(shard_chunks)):
            print(f"[SKIP] Shard {shard_id} is already complete and valid.")
            skipped_shards += 1
        else:
            shards_to_process.append((shard_idx, shard_chunks))

    if not shards_to_process:
        print("[OK] All selected shards are complete.")
        print(f"[OK] Skipped shards: {skipped_shards}")
        completed_count = (
            total_chunks
            if args.limit_shards is None
            else min(total_chunks, args.limit_shards * args.shard_size)
        )
        print(f"[OK] Completed records: {completed_count}")
        return

    print(
        f"[INFO] Initializing BGEM3FlagModel from pinned snapshot "
        f"{snapshot_path} for {len(shards_to_process)} shards..."
    )
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(
        str(snapshot_path),
        use_fp16=True,
    )

    processed_records_total = 0

    for shard_idx, shard_chunks in shards_to_process:
        shard_id = make_shard_id(model_id, config_hash, shard_idx)
        shard_dir = args.output_root / shard_id
        start_idx = shard_idx * args.shard_size
        end_idx = start_idx + len(shard_chunks)

        texts = [chunk["text_normalized"] for chunk in shard_chunks]

        outputs = model.encode(
            texts,
            batch_size=args.batch_size,
            max_length=MAXIMUM_MODEL_INPUT_TOKENS,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_embeddings = outputs["dense_vecs"]
        lexical_weights = outputs["lexical_weights"]

        records: list[ChunkEmbeddingRecord] = []

        for chunk, dense_vec, lex_weights in zip(
            shard_chunks, dense_embeddings, lexical_weights, strict=False
        ):
            dense_tuple = tuple(float(x) for x in dense_vec)
            sparse_vec = convert_sparse_vector(lex_weights)

            record = ChunkEmbeddingRecord(
                chunk_id=chunk["chunk_id"],
                parent_id=chunk["parent_id"],
                doc_id=chunk["doc_id"],
                dataset=chunk["dataset"],
                tokenizer_id=chunk["tokenizer_id"],
                model_id=model_id,
                model_revision=model_revision,
                embedding_config_hash=config_hash,
                dense_dimension=DENSE_DIMENSION,
                dense_precision=VectorPrecision.FLOAT32,
                dense_normalized=True,
                dense_vector=dense_tuple,
                sparse_vector=sparse_vec,
                source_chunk_sha256=chunk.get("text_sha256"),
            )
            records.append(record)

        header = EmbeddingShardHeader(
            dataset=DATASET_ID,
            shard_id=shard_id,
            shard_index=shard_idx,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_id="huggingface:BAAI/bge-m3@default:fast",
            embedding_kind=EmbeddingKind.HYBRID,
            dense_dimension=DENSE_DIMENSION,
            dense_precision=VectorPrecision.FLOAT32,
            dense_normalized=True,
            embedding_config_hash=config_hash,
            maximum_records=args.shard_size,
        )

        write_summary = write_embedding_shard(
            root=shard_dir,
            header=header,
            records=records,
            source_chunks_path=CHUNKS_PATH,
            source_chunks_manifest_path=CHUNKS_MANIFEST_PATH,
            overwrite=True,
            metadata={
                "dataset": DATASET_ID,
                "shard_id": shard_id,
                "shard_index": shard_idx,
                "synthetic": False,
                "semantic_embeddings": True,
                "source_start_index": start_idx,
                "source_end_index": end_idx,
            },
        )

        print(
            f"[OK] Shard {shard_id} written with {len(records)} records "
            f"range [{start_idx}:{end_idx}] -> {write_summary.paths.root.resolve()}"
        )
        processed_records_total += len(records)

    print(f"[OK] Total processed records in run: {processed_records_total}")


if __name__ == "__main__":
    main()

