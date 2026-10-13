# Pilot BGE-M3 CPU Embedding Benchmark and Materialization

**Step:** 83
**Status:** SUCCEEDED
**Scope:** Infrastructure validation only. Not a publishable result.

## Purpose

Perform the first approved loading of the locally cached BGE-M3 snapshot
and produce the first deterministic embedding artifacts for the pilot
synthetic dataset. No model download, no network access, no Qdrant
collection, no retrieval, no evaluation, no generation.

## Inputs

- Child chunks: `outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system/chunking/child_chunks.jsonl`
- Record count: 12
- Total input tokens: 1024 (min 76, median 87, max 94)

## Model

- Model: `BAAI/bge-m3`
- Revision: `5617a9f61b028005a4858fdac845db406aefb181` (frozen)
- Snapshot: `D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181`
- Cache status: VERIFIED
- Local files only: true
- Network fallback: false
- Device: CPU
- `use_fp16: false`, `use_bf16: false` (CPU-only host)
- Dense pooling: CLS
- Multi-vector (ColBERT): disabled

## Benchmark protocol

- Warmup runs: 1
- Measured runs per batch size: 3
- Batch sizes: 1, 2, 4
- Timings exclude model load time and persistence time

## Benchmark results

| Batch size | Raw runs (s) | Median (s) | Records/s | ms/record |
|-----------:|-------------:|-----------:|----------:|----------:|
| 1 | 10.470 / 10.476 / 10.498 | 10.476 | 1.145 | 873.0 |
| 2 | 10.228 / 10.203 / 10.189 | 10.203 | 1.176 | 850.2 |
| 4 | 11.903 / 11.918 / 11.934 | 11.918 | 1.007 | 993.2 |

- Selected batch size: **2**
- Selection reason: highest tested batch size that completed without
  failure and had median throughput no worse than the smaller
  successful batch sizes.
- Peak process RSS: ~1.9 GiB (only 5.9 GiB available on host).

## Materialization

- Second encode pass performed for determinism verification
- Shard writer: `rag_system.embedding.shards.write_embedding_shard`
- Output root: `outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system/embedding/`
- Files produced:
  - `shards/shard-00000/header.json`
  - `shards/shard-00000/records.jsonl`
  - `shards/shard-00000/shard.manifest.json`
  - `embedding.catalog.json`
  - `embedding.manifest.json`
  - `checksums.json`
- Records: 12
- Dense vectors: 12 (dim 1024, float32, L2-normalized, CLS pooling)
- Sparse vectors: 12 (learned lexical weights, indices ascending)
- Round-trip reload: valid
- Checksums: valid (SHA-256)
- **Determinism: BYTE_IDENTICAL**
  - max dense absolute difference: 0.0
  - max sparse absolute difference: 0.0
- Total output bytes: 293,269

## Execution state

| Flag | Value |
|---|---|
| model_loaded | true |
| benchmark_performed | true |
| embedding_materialization_performed | true |
| model_download_performed | false |
| network_access_performed | false |
| qdrant_execution_performed | false |
| retrieval_execution_performed | false |
| evaluation_execution_performed | false |
| generation_execution_performed | false |

## Non-publishability

- `publishable_result: false`
- `included_in_final_comparison: false`
- `runtime_measurement_for_planning_only: true`
- `retrieval_effectiveness_claimed: false`

Runtime on 12 records is not a valid dataset-level estimate and must
not be linearly extrapolated without token-count adjustment.

## Next gate

**Step 84: Qdrant Local-Persistence Ingestion Plan and Collection Freeze**

No movement to the seven real datasets until the full synthetic chain
is verified end-to-end:
canonical -> parser -> chunker -> embedding -> Qdrant -> retrieval -> evaluation