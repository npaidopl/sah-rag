# rag_system End-to-End Static Inspection

**Status:** Step 79 static inspection
**Result:** ADAPTER_REQUIRED, pilot-eligible after adapter.

## Scope

Read-only inspection of the project-owned architecture
`rag_system` (project `rag-architecture`, "Structure-Aware
Hierarchical Hybrid RAG"). No index, embedding, retrieval, model,
evaluation, or generation call is made.

## Source location

- Root: `src/rag_system`
- 99 files, 96 Python, 3 `py.typed` markers
- 13 subpackages (listed in `baseline_audit/my_rag_end_to_end_inspection.yaml`)
- Python `>=3.12,<3.13`, Pydantic 2.x, MIT license

## Offline stage map

| Stage | Executable | Implementation |
|---|---|---|
| canonical input validation | yes | `canonical.loaders`, `canonical.inspection` |
| derived preprocessing | partial | inline in parsing |
| structural parsing | yes | `parsing.pipeline.parse_canonical_dataset` |
| semantic parent construction | partial | parsing + chunking |
| hierarchical child chunking | yes | `chunking.hierarchical.HierarchicalChildChunker` |
| dense + sparse embedding | yes | `embedding.shards.write_embedding_shard` |
| vector-store ingestion | yes | `storage.qdrant.ingest_embedding_records` |
| parent + document storage | yes | Qdrant payload + embedding catalog |

## Online stage map

| Stage | Executable | Implementation |
|---|---|---|
| canonical query loading | yes | `canonical.loaders`, `evaluation.clerk` |
| dense query embedding | yes | `FlagEmbedding.BGEM3FlagModel` |
| sparse query embedding | yes | same model, lexical weights |
| dense retrieval | yes | `retrieve_dense` |
| sparse retrieval | yes | `retrieve_sparse` |
| rank fusion | yes | `retrieve_hybrid_rrf` (RRF) |
| top-k child retrieval | yes | `--child-candidate-limit` default 200 |
| semantic-parent resolution | yes | `aggregate_chunks_to_parents` |
| canonical-document resolution | yes | via `doc_id` in payload |
| evaluation | yes | `evaluate_retrieval_run` + CLERK adapter |

## Chunking

`HierarchicalChildChunker` v0.3.0. 256 model-input tokens per chunk,
32-token overlap, parent boundaries preserved (mandatory),
deterministic.

## Embedding

- Provider: FlagEmbedding (local)
- Dense model: `BAAI/bge-m3`, 1024-dim
- Sparse: BGE-M3 learned sparse lexical weights
- Normalized: yes
- Batch size: 2 (default), 4 (CLERK runs)
- Device: `cpu` on current host
- Isolated env: `.venv-embed` for the FlagEmbedding stack
- HF cache: `HUGGINGFACE_HUB_CACHE=D:\rag_work\hf_cache\hub`

**Revision is UNPINNED.** Each run must specify `model_revision`
in `EmbeddingConfig` for reproducibility.

## Storage

Qdrant, hybrid collection with named `dense` (COSINE) and `sparse`
(IP) vectors. Both local persistence mode and server mode are
supported. Point ID is deterministic:
`uuid5(NAMESPACE_URL, "rag-chunk:{chunk_id}")`.

The collection manifest is checksum-bound and contains the embedding
catalog SHA, config hash, model identity, and expected point count.

## Retrieval

- Fusion: RRF
- Default child candidate limit: 200
- Default document limit: 100
- Score direction: higher is better
- Reranker: optional
- `canonical_doc_ids` exposed via `doc_id` in payload

## Generation boundary

Retrieval is fully separable from generation.
`run_clerk_test_retrieval_evaluation.py` runs dense + sparse +
hybrid-RRF retrieval and evaluation with **no LLM call**. Generation
subsystem exists but is optional.

## Canonical envelope compatibility

**Status: ADAPTER_REQUIRED.**

The real `rag_system` envelope is:
processed/canonical/<dataset>/
documents.jsonl
examples.jsonl
validation.json
manifest.json
The Step 78 synthetic pilot envelope is:
data/pilot/infrastructure-pilot-synthetic-v1/
documents.jsonl
chunks.jsonl
queries.jsonl
judgments.jsonl
dataset.manifest.json

The adapter must:

1. Merge `queries.jsonl` + `judgments.jsonl` into `examples.jsonl`
   using the `CanonicalExample` shape (`query` + `ground_truth`).
2. Emit `manifest.json` in the shape expected by
   `canonical.paths`.
3. Generate `validation.json`.
4. Either consume `chunks.jsonl` as-is (rewrapping into `ChildChunk`)
   or regenerate from `documents.jsonl` and compare.

## Host feasibility

- CPU-only supported: yes
- Mandatory GPU: no
- Mandatory Qdrant server: no (local persistence mode)
- Mandatory hosted LLM: no
- Mandatory secrets: no
- Estimated CPU wall-clock for 24-chunk pilot: 1--3 minutes

## Pilot eligibility

Eligible **after** the adapter exists. Execution mode:
retrieval-only. Prohibited: generation, hosted LLM calls, foreign
baseline execution.

## Prohibited claims

- Static inspection is not an execution result.
- Compatibility assessment is not retrieval validation.
- CPU feasibility is not runtime benchmarking.
- No index, embedding, retrieval, evaluation, or generation ran.

## Next gate

**Step 80 — Canonical Envelope Adapter.** Only after that adapter is
validated do we read the seven real datasets and size their embedding
time.
