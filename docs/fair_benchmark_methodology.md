# Fair Benchmark Methodology

**Step:** 93
**Status:** FROZEN_FOR_IMPLEMENTATION
**Scope:** Methodological checkpoint. No execution.

## Purpose

Freeze the methodological rules that must hold before any of the
eleven architectures are adapted or executed. This is planning-only:
no pipeline runs, no tokenizer loads, no model loads, no embeddings,
no retrieval, no evaluation.

## The 11 architectures

- `rag_system` (project-owned reference)
- `longrag`, `raptor`, `hipporag2`, `lightrag`, `pathrag`
- `disretrieval`, `memorag`, `kag`, `metadata_rag`, `hikey`

## Two benchmark tracks

### Track A --- Controlled Architectural

Primary comparison. Common components:

- dataset suite: `canonical-rag-dataset-suite`
- embedder: `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181`
- evaluator: common retrieval evaluation contract
- generation: disabled

Architecture-specific components preserved:
preprocessing, hierarchy construction, graph construction,
candidate generation, fusion, traversal, aggregation, expansion.

**Prohibited claim**: faithful reproduction of every original system.

### Track B --- Native-Fidelity Reference

Small stratified subset where each architecture runs with its
native components. Purpose: quantify how much controlled
substitution changes each architecture.

**Not leaderboard-eligible. Not merged with Track A scores.**

### Track C --- Deferred Appendix

Architectures that cannot be executed under either track appear as
feasibility and coverage appendix entries, **not** numeric
leaderboard rows.

## Three status concepts per architecture

- **implementation_status**: NOT_STARTED / PARTIAL /
  SYNTHETIC_VALIDATED / DATASET_READY
- **fidelity_status**: NATIVE_FAITHFUL / CONTROLLED_ADAPTATION /
  PAPER_DERIVED_COMPARATOR / UNRESOLVED
- **benchmark_eligibility**: PRIMARY_CONTROLLED_TRACK /
  SECONDARY_NATIVE_TRACK / BOTH_TRACKS / DEFERRED / EXCLUDED_WITH_REASON

## Component taxonomy

| Role | Replacement default |
|---|---|
| ARCHITECTURE_DEFINING | PROHIBITED |
| ARCHITECTURE_SUPPORTING | ALLOWED_WITH_EQUIVALENCE_VALIDATION |
| OPTIONAL_DOWNSTREAM | DISABLE_FOR_RETRIEVAL_ONLY |

## Storage substitution policy

**Qdrant allowed** when the storage provides only dense/sparse
nearest-neighbor search, distance metric is preserved, filters are
preserved, and top-k equivalence validates.

**Native storage required** when graph traversal occurs inside
storage, database-native reasoning is architecture-defining, or
backend changes candidate generation.

Required equivalence test per Qdrant substitution: minimum 10
fixture queries, compare top-10 overlap, top-100 overlap, exact
point identity, score order, filter behavior.

## LLM policy

Final generation: disabled (retrieval-only benchmark).

Index-time and query-time LLM stages: preserved when
architecture-defining, with deterministic configuration
(model_id, revision, temperature=0, seed, prompt_sha256,
output_schema, retry_policy, cost_report).

## Precision equivalence protocol

Pre-registered thresholds:

- median same-record cosine >= 0.999
- p01 same-record cosine >= 0.995
- mean top-10 overlap >= 0.99
- mean top-100 overlap >= 0.99
- absolute recall@10 delta <= 0.005
- absolute ndcg@10 delta <= 0.005

Decision rule: pass -> FP16 production; fail -> FP32 production.

## Statistical analysis plan

- per-dataset: macro-average over queries
- cross-dataset: macro-average over datasets
- global micro-average: reported, not primary
- uncertainty: paired bootstrap, 10000 resamples, seed=0
- multiple testing: HOLM correction
- context-length bins frozen before quality results

## Dataset retrieval task matrix

Each dataset has an explicit query text projection, permitted
corpus selector, primary retrieval unit, primary evaluation unit,
zero-relevance policy, and leakage prohibitions.

**Supplied context is never automatically embedded as query text.**

## CUAD evaluation policy

Two subsets reported separately:

- positive evidence (6702 queries): hit_at_k, recall_at_k,
  reciprocal_rank, ndcg_at_k
- zero evidence (14208 queries): false_positive_retrieval_rate,
  zero_result_rate, minimum_score_distribution

**Macro mixing of the two subsets is prohibited.**

## Hyperparameter policy

`rag_system` tuning: requires development data, prohibits test
data for tuning, freezes selected configuration before test.

Baselines: paper/author/repository defaults, or equal small budget
tuning when defaults unavailable. **Unequal extensive tuning
prohibited.**

## Next gate

**Step 94: Per-Architecture Readiness Gap Registry.**

Then per-architecture one at a time: contract -> implementation ->
tiny synthetic proof -> persistence -> retrieval -> canonical
evaluation mapping -> readiness declaration.

Proposed order: PathRAG comparator, LongRAG, RAPTOR, LightRAG,
HippoRAG 2, Metadata-RAG, DISRetrieval, MemoRAG, KAG, HiKEY.