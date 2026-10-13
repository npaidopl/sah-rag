# SAH-RAG M7 β€” Locked Test Evaluation

**Frozen config**: `c256_o32_hybrid_rrf_k200`

- chunk_size = 256
- chunk_overlap = 32
- retrieval_mode = hybrid_rrf
- candidate_limit = 200

## Per-dataset results

| Dataset | Queries | H@10 | R@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| clerk | 587 | 0.5332 | 0.3626 | 0.2936 | 0.3348 |
| cuad | 10455 | 0.0122 | 0.0122 | 0.0055 | 0.0035 |
| greek_bar_bench | 142 | 0.2746 | 0.2746 | 0.1282 | 0.0841 |
| lara | 1139 | 0.9350 | 0.9350 | 0.8432 | 0.8135 |
| **MACRO** | | 0.4388 | 0.3961 | 0.3177 | 0.3090 |

## Primary metric

`macro_document_recall@10 = 0.396121`

## Full metric set (macro)

- `hit@1` = 0.254512
- `hit@10` = 0.438785
- `hit@20` = 0.525080
- `hit@5` = 0.382391
- `hit@50` = 0.628874
- `mrr@1` = 0.254512
- `mrr@10` = 0.308981
- `mrr@20` = 0.315035
- `mrr@5` = 0.301691
- `mrr@50` = 0.318295
- `ndcg@10` = 0.317657
- `ndcg@50` = 0.363760
- `precision@10` = 0.053546
- `recall@1` = 0.218028
- `recall@10` = 0.396121
- `recall@20` = 0.480713
- `recall@5` = 0.340435
- `recall@50` = 0.586774

## Protocol

Locked test: single execution of the frozen config on the test split. No parameter adjustment was performed after reading test metrics.