# Metadata-RAG vs Baselines — Comparison

Retrieval-only comparison of `Metadata-RAG` against `LongRAG` and `My_RAG`.

**Primary metric:** macro document `recall_at_10` across the four datasets.

## 1. Per-dataset primary metric

| Dataset | Metadata-RAG | LongRAG | My_RAG |
|---|---|---|---|
| clerk | 0.2726 | 0.0786 | 0.3546 |
| cuad | 0.0294 | 0.0324 | 0.0431 |
| greek_bar_bench | 0.2711 | 0.2606 | 0.2852 |
| lara | 0.8874 | 0.8861 | 0.9368 |

## 2. Secondary macro across datasets

| Metric | Metadata-RAG | LongRAG | My_RAG |
|---|---|---|---|
| Hit@1 | 0.2313 | 0.1957 | 0.2553 |
| Hit@5 | 0.3389 | 0.2606 | 0.3846 |
| Hit@10 | 0.4085 | 0.3183 | 0.4520 |
| Hit@20 | 0.4789 | 0.3881 | 0.5416 |
| Hit@50 | 0.5920 | 0.4929 | 0.6658 |
| Hit@100 | 0.6570 | — | — |
| Precision@10 | 0.0482 | 0.0362 | 0.0557 |
| Recall@1 | 0.2026 | 0.1905 | 0.2180 |
| Recall@5 | 0.2999 | 0.2569 | 0.3394 |
| Recall@10 | 0.3651 | 0.3144 | 0.4049 |
| Recall@20 | 0.4316 | 0.3833 | 0.4938 |
| Recall@50 | 0.5478 | 0.4870 | 0.6210 |
| Recall@100 | 0.6121 | — | — |
| MRR@10 | 0.2797 | 0.2268 | 0.3123 |
| MRR@50 | 0.2882 | 0.2349 | 0.3225 |
| nDCG@10 | 0.2883 | 0.2476 | 0.3214 |
| nDCG@50 | 0.3319 | 0.2880 | 0.3733 |

## 3. Dataset coverage

| Track | Present | Missing |
|---|---|---|
| Metadata-RAG | clerk, cuad, greek_bar_bench, lara | — |
| LongRAG | clerk, cuad, greek_bar_bench, lara | — |
| My_RAG | clerk, cuad, greek_bar_bench, lara | — |

## 4. Controlled adaptations

- All tracks are retrieval-only; generation is disabled.
- Metadata-RAG uses the frozen clean-room spec plus amendments 01-04.
- Native-fidelity track is DEFERRED_WITH_REASON for Metadata-RAG.

