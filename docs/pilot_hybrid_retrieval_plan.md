# Pilot Hybrid Retrieval Plan

**Step:** 86
**Status:** PLANNED (planning only)
**Scope:** Infrastructure validation only. Not a publishable result.

## Purpose

Freeze the retrieval boundary for the pilot synthetic dataset:
six queries, query-embedding identity, dense/sparse query vectors,
Qdrant collection identity, candidate/result limits, RRF fusion
boundary, ranked-item persistence contract, parent aggregation
contract, document resolution contract, and the acceptance gates
of the eventual Step 87 execution.

The following stages remain disabled:

- query model loading
- query embedding generation
- dense retrieval execution
- sparse retrieval execution
- hybrid retrieval execution
- parent aggregation execution
- document expansion execution
- evaluation
- generation

## Query source

- Path: `data/pilot/infrastructure-pilot-synthetic-v1/queries.jsonl`
- Records: 6
- Query IDs: `pilot-query-01` ... `pilot-query-06`
- Canonical correspondence:
  `processed/canonical/infrastructure-pilot-synthetic-v1/examples.jsonl`

The planner never uses `reference_answer`, `relevance`, `judgment`,
or `relevance_label`. The `target_content_group_id` field is
present but explicitly marked as not used for filtering or ranking.

## Collection

- Physical name: `rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0`
- Backend: Qdrant, mode local persistence
- Points: 12 (validated in Step 85)

## Query embedding contract

- Model: `BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181`
- Device: CPU, local files only
- Dense: 1024, normalized, CLS pooling
- Sparse: learned lexical weights
- Multi-vector (ColBERT): disabled
- Batch size: 2 (consistent with Step 83 selection)

## Retrieval mode

- Primary: `hybrid_rrf`
- Diagnostic components: `dense`, `sparse`
- Reranker: disabled

### RRF implementation

The project-owned `retrieve_hybrid_rrf` delegates fusion to Qdrant
via `models.FusionQuery(fusion=models.Fusion.RRF)`. No project-side
constant is set.

- **Implementation:** `QDRANT_NATIVE_RRF`
- **Rank constant:** `UNPINNED` (no source evidence for a specific k)
- **Tie-break:** `QDRANT_DEFAULT`

Step 87 must not claim a specific rank constant.

## Limits

- Candidate limit: 12
- Child result limit: 12
- Parent result limit: 12
- Document result limit: 12

All limits are exactly the collection size. Requested evaluation
cutoffs (1, 5, 10, 100) are recorded but not padded: the ranked
result list uses only real returned items.

## Parent aggregation (frozen from real code)

- Function: `rag_system.retrieval.aggregation.aggregate_chunks_to_parents`
- Method: `MAX_SCORE_PLUS_WEIGHTED_ADDITIONAL_CHILD_SCORE`
- `additional_child_weight` = 0.05
- Algorithm:
  - group ranked children by `parent_id`
  - `best_score = max(child.score)` per group
  - `additional_score = sum(child.score)` of the remaining children
  - `parent_score = best_score + 0.05 * additional_score`
  - `best_chunk_rank` = smallest child rank in the group
- Parent tie-break (from source):
  - `parent_score DESC`
  - `best_chunk_rank ASC`
  - `parent_id ASC`

## Document resolution

There is **no** typed `RetrievedDocument` model in the codebase at
Step 86. Document resolution is defined by this contract as a
project-owned mapping:

- Method: `MAX_PARENT_SCORE_PER_DOCUMENT`
- Document identity: `RetrievedParent.doc_id`
- Document score: score of the highest-ranked parent of that doc
- Contributing parent and child IDs retained
- Duplicate-content documents are **not** deduplicated
- Document tie-break (frozen by this contract):
  - `document_score DESC`
  - `best_parent_rank ASC`
  - `doc_id ASC`

## Output plan

Root: `outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system/retrieval`

Planned files (produced only in Step 87):

- `query_embeddings.jsonl`
- `ranked_children.jsonl`
- `ranked_parents.jsonl`
- `ranked_documents.jsonl`
- `retrieval.manifest.json`
- `checksums.json`

Serialization is UTF-8, LF line endings, compact JSON, sorted keys,
atomic writes, SHA-256 checksums.

## Acceptance gates (Step 87)

- exact query count 6, unique IDs
- collection point count 12
- query embeddings valid
- dense/sparse/hybrid results present
- child/parent/document ranks valid and contiguous
- lineage complete across all three artifact levels
- duplicate document identity preserved
- evaluation/generation execution false

## Next gate

**Step 87: bounded hybrid retrieval execution for exactly six pilot
queries.** The evaluation stage remains disabled.