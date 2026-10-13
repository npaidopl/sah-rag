# Pilot Bounded Hybrid Retrieval Execution

**Step:** 87
**Status:** APPROVED_FOR_BOUNDED_EXECUTION
**Scope:** Infrastructure validation only. Not a publishable result.

## Purpose

Execute the first bounded hybrid retrieval on the pilot synthetic
dataset: encode six queries with the frozen BGE-M3 snapshot, run
dense / sparse / hybrid RRF retrieval on the persisted Qdrant
collection, aggregate children to parents, resolve parents to
canonical documents, and persist deterministic retrieval artifacts.

Evaluation and generation remain disabled. Judgments and reference
answers are never loaded.

## Inputs

- Queries: `data/pilot/infrastructure-pilot-synthetic-v1/queries.jsonl` (6)
- Canonical documents: 12 (referenced only for lineage)
- Qdrant collection:
  `rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0` (12 points)
- Model: `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`

## Query leakage guard

The runner loads only `query_id` and `query_text` into the runtime
context. The following are never loaded or referenced in runtime:

- `judgments.jsonl`
- `target_content_group_id`
- `relevance` / `relevance_label`
- `reference_answer`
- `ground_truth`

## Retrieval

- Primary: `hybrid_rrf` (Qdrant-native RRF)
- Diagnostic components: `dense`, `sparse`
- `candidate_limit`: 12
- `child_result_limit`: 12
- `parent_result_limit`: 12
- `document_result_limit`: 12
- Reranker: disabled

## Parent and document resolution

- Parent aggregation: `aggregate_chunks_to_parents`
  (`MAX_SCORE_PLUS_WEIGHTED_ADDITIONAL_CHILD_SCORE`, weight 0.05)
- Parent tie-break: `(parent_score DESC, best_chunk_rank ASC, parent_id ASC)`
- Document resolution: `resolve_parents_to_documents`
  (`MAX_PARENT_SCORE_PER_DOCUMENT`)
- Document tie-break: `(document_score DESC, best_parent_rank ASC, doc_id ASC)`
- Duplicate-content documents are preserved

## Output

Root: `outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system/retrieval`

- `query_embeddings.jsonl`
- `ranked_children.jsonl`
- `ranked_parents.jsonl`
- `ranked_documents.jsonl`
- `retrieval.manifest.json`
- `checksums.json`

Serialization is UTF-8, LF line endings, compact JSON, sorted keys,
SHA-256 checksums.

## Immutability guarantees

- Embedding artifacts must be byte-identical before/after
- Qdrant collection must remain 12 points, same point IDs, same
  payloads, same schema (semantic immutability)
- No vector was inserted or modified

## Acceptance gates

- exact query count 6
- collection point count 12 (before and after)
- query embeddings valid (dim 1024, finite, L2-normalized, sparse
  indices sorted/unique/non-negative)
- dense / sparse / hybrid results present for every query
- contiguous ranks, unique IDs per query/method
- complete lineage across children, parents, documents
- duplicate document identity preserved
- round trip valid, checksums valid
- evaluation / generation execution false

## Next gate

**Step 88: common retrieval evaluation plan for six pilot queries.**

Evaluation itself remains disabled until the next step explicitly
authorizes it.