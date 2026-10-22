"""Create a synthetic embedding shard from sample chunks."""

from __future__ import annotations

from pathlib import Path

from sah_rag.artifacts import (
    ChildChunk,
    sha256_fingerprint,
)
from sah_rag.canonical.jsonl import iter_jsonl
from sah_rag.embedding import (
    EmbeddingKind,
    EmbeddingShardHeader,
    VectorPrecision,
    build_embedding_shard_id,
    create_synthetic_embedding,
    write_embedding_shard,
)

CHUNKS_PATH = Path(
    "outputs/clerk_chunks_auto.jsonl"
)

CHUNKS_MANIFEST_PATH = Path(
    "outputs/clerk_chunks_auto.manifest.json"
)

SHARD_ROOT = Path(
    "outputs/synthetic_embedding_shard_000000"
)


def main() -> None:
    """Create deterministic synthetic hybrid embeddings."""

    chunks = [
        ChildChunk.model_validate(record)
        for record in iter_jsonl(CHUNKS_PATH)
    ]

    embedding_configuration = {
        "embedder_id": "synthetic_test_embedder",
        "embedder_version": "0.1.0",
        "model_id": "synthetic/bge-m3-shape",
        "model_revision": "synthetic-v1",
        "tokenizer_id": (
            "huggingface:BAAI/bge-m3@default:fast"
        ),
        "embedding_kind": "hybrid",
        "dense_dimension": 16,
        "dense_precision": "float32",
        "dense_normalized": True,
        "synthetic": True,
    }

    config_hash = sha256_fingerprint(
        embedding_configuration
    )

    shard_id = build_embedding_shard_id(
        dataset="clerk",
        model_id=embedding_configuration[
            "model_id"
        ],
        embedding_config_hash=config_hash,
        shard_index=0,
    )

    header = EmbeddingShardHeader(
        shard_id=shard_id,
        shard_index=0,
        dataset="clerk",
        model_id=embedding_configuration[
            "model_id"
        ],
        model_revision=embedding_configuration[
            "model_revision"
        ],
        tokenizer_id=embedding_configuration[
            "tokenizer_id"
        ],
        embedding_config_hash=config_hash,
        embedding_kind=EmbeddingKind.HYBRID,
        dense_dimension=16,
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=True,
        maximum_records=1000,
    )

    records = (
        create_synthetic_embedding(
            chunk=chunk,
            model_id=header.model_id,
            model_revision=header.model_revision,
            tokenizer_id=header.tokenizer_id,
            embedding_config_hash=(
                header.embedding_config_hash
            ),
            dense_dimension=16,
        )
        for chunk in chunks
    )

    summary = write_embedding_shard(
        root=SHARD_ROOT,
        header=header,
        records=records,
        source_chunks_path=CHUNKS_PATH,
        source_chunks_manifest_path=(
            CHUNKS_MANIFEST_PATH
        ),
        overwrite=True,
        execution_profile="local_cpu",
        metadata={
            "synthetic": True,
            "semantic_embeddings": False,
        },
    )

    print("[OK] Synthetic shard")
    print("  Shard ID:", summary.shard_id)
    print("  Records:", summary.record_count)
    print("  Header:", summary.paths.header.resolve())
    print("  Records file:", summary.paths.records.resolve())
    print("  Manifest:", summary.paths.manifest.resolve())
    print(
        "  WARNING: vectors are synthetic and "
        "must not be used for retrieval evaluation."
    )


if __name__ == "__main__":
    main()
