# sah_rag.embedding -- Infrastructure Module

**Status:** FROZEN as of 2026-11-04 (FIX-03 / F4)

## Purpose

This module provides typed, deterministic infrastructure for embedding
artifacts in the SAH-RAG project:

- Pydantic models (`EmbeddingConfig`, `ChunkEmbeddingRecord`, `SparseVector`,
  `EmbeddingShardHeader`, `EmbeddingRunCatalog`)
- Atomic shard writing (`write_embedding_shard`)
- Cross-shard validation (catalog consistency, shard index continuity,
  duplicate chunk detection)
- Synthetic embeddings for tests (`create_synthetic_embedding`)

## What this module DOES NOT provide

**The actual embedder is NOT in this module.**

The production BGE-M3 embedder lives in:

- `scripts/colab_sah_rag_m0_embed_chunks.py` -- chunk embeddings (Colab GPU)
- `scripts/colab_sah_rag_m0_embed_queries.py` -- query embeddings (Colab GPU)
- `sah_rag/components/embedders.py::BGEM3Embedder` -- dense-only embedder
  used by controlled-mode baselines

This is intentional. The infrastructure module defines the *shape* of
the artifacts (Pydantic models, shards, manifests), not the *content*
(the actual model inference).

## Why the split?

- **Portability.** The infrastructure can be imported without pulling
  torch/transformers/FlagEmbedding (which are heavy GPU dependencies).
- **Testability.** Tests can build fake embedding artifacts via
  `create_synthetic_embedding` without loading any model.
- **Separation of concerns.** Model loading (GPU, CUDA, weights) is
  Colab-only. Artifact management (files, manifests, validation) is
  CPU-only.
## The two embedding config hashes

There are **two** different `embedding_config_hash` schemes in the
repository. This is intentional and is **not** a bug:

### Scheme 1: Colab production path

Used by `scripts/colab_sah_rag_m0_embed_chunks.py`.

    config_hash = sha256(
        f"{MODEL_ID}:{MODEL_REVISION}:hybrid:fp16:gpu:{chunker_hash}"
    ).hexdigest()

### Scheme 2: Pydantic infrastructure path

Used by `sah_rag.embedding.shards` when calling `write_embedding_shard`.

    config_hash = sha256_fingerprint(EmbeddingConfig.model_dump())
### Why two?

- Scheme 1 is minimal (matches what the Colab script actually cares about:
  model, revision, kind, precision, device, chunker).
- Scheme 2 is comprehensive (includes all `EmbeddingConfig` fields,
  making it forward-compatible if new fields are added).

The two are **not interoperable** by design. Artifacts produced by
Scheme 1 cannot be validated against Scheme 2 hashes, and vice versa.

### Implications

If you need to mix artifacts from both schemes, you **must** re-run
one of them. In practice:

- Production SAH-RAG pipeline uses **Scheme 1** (Colab).
- Baseline controlled-mode embedding uses **Scheme 1** (BGE-M3 dense-only
  via `sah_rag.components.embedders.BGEM3Embedder`, but artifacts still
  use the Colab hash scheme for consistency).
- Pydantic infrastructure (**Scheme 2**) is used **only** for synthetic
  tests and future artifacts.
## Models

- EmbeddingConfig -- frozen; captures model_id, model_revision,
  tokenizer_id, embedding_kind, dense_dimension, dense_precision,
  dense_normalized, device, use_fp16, batch_size,
  maximum_model_input_tokens, shard_maximum_records
- ChunkEmbeddingRecord -- one chunk's embedding (dense + sparse +
  provenance)
- SparseVector -- indices (sorted, unique, >=0) + values (finite)
- EmbeddingShardHeader -- shard metadata (identity, config, record count)
- EmbeddingRunCatalog -- full run manifest (shard continuity, coverage)

## Functions

- write_embedding_shard() -- atomic write with sidecar manifest + rollback
- create_synthetic_embedding() -- deterministic non-semantic test vector
- build_embedding_config_hash() -- Pydantic-style hash (Scheme 2)
- validate_shard_against_catalog() -- cross-shard consistency check
## Testing

Synthetic embeddings allow offline testing:

    from sah_rag.embedding import create_synthetic_embedding

    vec = create_synthetic_embedding(dimension=1024, seed=42)
    assert len(vec) == 1024

## References

- Shard infrastructure docs: docs/embedding_shards.md
- v1 vs v2 evaluation boundary: docs/evaluation_boundary_v1_v2.md
- Model config: sah_rag/embedding/config.py
- Shard writing: sah_rag/embedding/shards.py
- Catalog: sah_rag/embedding/catalog.py

## Change log

- 2026-11-04 (FIX-03 / F4): documented the two-hash-scheme design
  and the module/infrastructure split.