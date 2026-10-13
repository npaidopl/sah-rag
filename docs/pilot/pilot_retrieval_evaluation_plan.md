# Pilot Retrieval Evaluation Plan

**Step:** 88
**Status:** PLANNED (planning only)
**Scope:** Infrastructure validation only. Not a publishable result.

## Purpose

Freeze how the six ranked-document lists of Step 87 connect to the
twelve pilot judgments, and how they will be evaluated via the
frozen common retrieval evaluation contract. This step is
planning-only: no evaluator call, no metric computation, no
evaluation output directory.

## Evaluation target

- Primary: `hybrid_rrf` → ranked parents → ranked canonical documents
- Diagnostic-only (not separate architecture comparisons):
  `dense`, `sparse` child results

## Sources

- Ranked documents:
  `outputs/infrastructure_pilot/.../rag_system/retrieval/ranked_documents.jsonl`
- Judgments:
  `data/pilot/infrastructure-pilot-synthetic-v1/judgments.jsonl`

### Inventory

- Queries: 6 (`pilot-query-01` .. `pilot-query-06`)
- Judgments: 12 (2 per query)
- Relevance: binary, relevant value = 1
- Item type: `canonical_document`
- Retrieved identity field: `doc_id`

## Evaluation unit

`canonical_document`. Child chunks, semantic parents, and content
groups are not evaluated directly.

## Duplicate-content policy

The two canonical documents per content group remain distinct
relevant items. Content-based deduplication is prohibited; a
retrieved document is not treated as a hit for its duplicate just
because they share a content checksum.

## Cutoffs

Frozen from the common contract: `[1, 5, 10, 100]`.

The ranked-document list contains at most 12 documents per query.
The evaluation evaluates the available prefix at cutoffs 1, 5, 10
and uses all available documents at cutoff 100. Synthetic padding
and missing-rank invention are prohibited.

## Metrics

From the common retrieval evaluation contract v0.1.0:

- `hit_at_k`
- `precision_at_k` (denominator = fixed_cutoff)
- `recall_at_k` (empty judgment → reject case)
- `reciprocal_rank` (cutoff applied)
- `ndcg_at_k` (gain `2^relevance - 1`, zero-IDCG → reject case)

Aggregation is macro-average with equal query weight.

No LLM judge, no answer matching, no generation metrics.

## Output plan

Root: `outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system/evaluation`

Planned files (produced only in Step 89):

- `retrieval_cases.jsonl`
- `query_metrics.jsonl`
- `aggregate_metrics.json`
- `evaluation.manifest.json`
- `checksums.json`

Serialization is UTF-8, LF, compact JSON, sorted keys, SHA-256
checksums.

## Prohibited outputs

- generated answers
- reference answer text
- answer correctness
- architecture ranking
- publishable comparison
- LLM judge results

## Leakage boundary

Judgments are available **only** to the evaluator. They are not
available to retrieval, query embedding, parent resolution, or
document resolution.

The planning step inspects judgments for validation and hash
computation only. The runtime evaluation runner (Step 89) is the
only component that may load them.

## Next gate

**Step 89: bounded common retrieval evaluation for six pilot
queries.** Evaluation runs against the hybrid_rrf ranked documents,
with no generation, no answer evaluation, and no publishable
architecture comparison.