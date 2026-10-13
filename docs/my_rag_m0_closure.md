# My_RAG (rag_system) M0 Baseline Closure Report

**Fidelity class:** PROJECT_OWNED_REFERENCE

## Scope

Datasets executed: clerk, cuad, greek_bar_bench, lara

## Per-dataset summary

| Dataset | Chunks | Queries | H@10 | R@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| clerk | 5317 | 1000 | 0.5430 | 0.3546 | 0.2875 |
| cuad | 30932 | 20910 | 0.0431 | 0.0431 | 0.0199 |
| greek_bar_bench | 23863 | 284 | 0.2852 | 0.2852 | 0.1314 |
| lara | 41702 | 2326 | 0.9368 | 0.9368 | 0.8466 |
| **MACRO** | | | 0.4520 | 0.4049 | 0.3214 |

## Primary metric

`macro_document_Recall_at_10 = 0.404936`

## Controlled adaptations

- **Qdrant → numpy retrieval**: mathematically equivalent in-memory computation (dense matmul + sparse csr + RRF k=60 + parent aggregation w=0.05 + document resolution).
- **Query dense re-exposure**: My_RAG dense query embeddings were regenerated alongside sparse (single Colab pass), not reused from LongRAG cache.

## Documented limitations

- Reduced dataset scope (4 of 7 datasets).
- FP16 embeddings, same precision as LongRAG.
- cuad: zero-relevance queries excluded per evaluator contract.
