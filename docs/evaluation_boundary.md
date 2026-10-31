# Evaluation Boundary - v2 (FROZEN)

**Metric version:** v2
**Status:** Frozen. Any change requires a version bump to v3 and migration of all reports.

---

## 1. Cutoffs

[1, 5, 10]

Previous version (v1) used [1, 3, 5, 10] without hit@k. v2 aligns with BEIR / MTEB / LongRAG / DISRetrieval / RAPTOR conventions.

## 2. Metrics (per cutoff k in {1, 5, 10})

| Metric    | Definition |
|-----------|-----------|
| Recall@k  | |retrieved[:k] intersect gt| / |gt| |
| MRR@k     | 1 / rank of first gt doc in retrieved[:k], else 0 |
| nDCG@k    | DCG@k / IDCG@k with binary relevance |
| Hit@k     | 1.0 if any gt doc in retrieved[:k], else 0.0 |

All metrics operate at document level. Chunks are aggregated to their parent doc_id before metric computation.

---
## 3. Averaging

Both reported per run:

- **Macro:** mean over queries (equal weight per query), then mean over datasets (equal weight per dataset).
- **Micro:** pooled over all queries across all datasets (each query contributes equally regardless of dataset).

**v1 to v2 incompatibility:** macro averages are NOT comparable across metric versions. Any historical table must be recomputed or explicitly labeled as v1.

## 4. Exclusions

Queries with empty gt_doc_ids are excluded from all metrics. The count of excluded queries is recorded per run in metrics.json under n_excluded.

---

## 5. Splits

- If a dataset provides an official test split, only test is evaluated.
- Otherwise, the full dataset is evaluated (no random splits, no leakage).

## 6. Reproducibility

- Seed: 42 for any stochastic operation (e.g., clustering in RAPTOR).
- Retrieval order: deterministic tie-breaks by chunk_id ascending when scores are equal.
- All runs write a config.json snapshot of inputs.

---

## 7. Output schema (metrics.json)

Fields: run_id, baseline, dataset, mode, metric_version="v2", cutoffs=[1,5,10], n_examples, n_evaluated, n_excluded, metrics={recall@1/5/10, mrr@1/5/10, ndcg@1/5/10, hit@1/5/10}.

## 8. Downstream consumers

Any consumer that reads metrics.json MUST check metric_version and handle unknown versions gracefully. Consumers that assume [1, 3, 5, 10] or hit@3 must be updated.
