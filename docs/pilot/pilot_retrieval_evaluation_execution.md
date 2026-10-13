# Pilot Bounded Common Retrieval Evaluation

**Step:** 89
**Status:** APPROVED_FOR_BOUNDED_EXECUTION
**Scope:** Infrastructure validation only. Not a publishable result.

## Purpose

Execute the first bounded common retrieval evaluation on the six
pilot queries, using the frozen hybrid_rrf ranked canonical
documents of Step 87 and the frozen judgment mapping of Step 88.
Never runs retrieval, never loads the model, never opens Qdrant,
never executes generation or LLM-as-judge.

## Inputs

- Ranked documents:
  `outputs/infrastructure_pilot/.../rag_system/retrieval/ranked_documents.jsonl`
- Judgments:
  `data/pilot/infrastructure-pilot-synthetic-v1/judgments.jsonl`
- Retrieval execution manifest:
  `baseline_audit/pilot_retrieval_execution_manifest.yaml` (SUCCEEDED)
- Judgment plan hash: `717d981d0f10ecffec81ed026e2e535e8c8434d8f2917d38919596e238ea76f6`
- Evaluation plan hash: `7a1580ba5f9f9e276e8d621cf0ed321d522af61bcf69e53086f2295ff66ba592`

## Evaluation unit

`canonical_document`. Retrieved identity field: `doc_id`.
Judgment identity field: `item_id`.

## Metric row model (typed)

The evaluator persists typed rows, not normalized per-metric rows:

- `QueryCutoffMetrics` --- one row per `(query_id, cutoff)`
  - **6 queries × 4 cutoffs = 24 rows**
- `AggregateCutoffMetrics` --- one row per `cutoff`
  - **4 rows**
- Each row carries all five metrics simultaneously.

## Metrics

Frozen from `common_retrieval_evaluation_contract.yaml` v0.1.0:

- `hit_at_k`
- `precision_at_k` (fixed_cutoff denominator)
- `recall_at_k` (reject_case when no relevant item)
- `reciprocal_rank` (cutoff applied)
- `ndcg_at_k` (2^rel-1 gain, zero-IDCG → reject_case)

Macro-average with equal query weight.

## Cutoffs

`[1, 5, 10, 100]`.

The ranked list contains at most 12 documents per query. Cutoff 100
uses the available 12 with fixed_cutoff denominator: precision@100 =
relevant_retrieved_within_12 / 100. This is expected under the
frozen common contract and is not altered for the pilot.

## Output

Root: `outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system/evaluation`

- `retrieval_cases.jsonl`
- `query_metrics.jsonl`
- `aggregate_metrics.json`
- `evaluation.manifest.json`
- `checksums.json`

## Prohibited outputs

- answers.jsonl
- predictions.jsonl
- llm_judge.jsonl
- architecture_ranking.json
- comparison_table.json

## Immutability

Before/after execution, retrieval artifacts and judgments must be
byte-identical. No Qdrant or embedding access occurs.

## Determinism

Two runs on the same frozen inputs produce byte-identical data
artifacts because:

- inputs are frozen
- rank lists are frozen
- judgments are frozen
- metric functions are deterministic
- serialization is canonical JSON

## Acceptance gates

- exact query count 6
- exact judgment count 12
- exact retrieval case count 6
- exact query metric row count 24
- exact aggregate metric row count 4
- every query present, every cutoff present
- all metric values finite and in [0, 1]
- macro means match per-query means (tolerance 1e-12)
- duplicate document identity preserved
- round trip valid, checksums valid
- retrieval, generation, answer evaluation, LLM judge, architecture
  comparison all disabled

## Next gate

**Step 90: infrastructure pilot closure and evidence freeze.**