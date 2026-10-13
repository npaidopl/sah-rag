# SAH-RAG Optimal-Operation Study β€” Closure

**Status:** SAH_RAG_OPTIMAL_OPERATION_STUDY_CLOSED

## Selected configuration

- `c256_o32_hybrid_rrf_k200`

## Primary metric

| Split | Macro R@10 |
|---|---:|
| Dev (selection) | 0.4068 |
| Test (locked)   | 0.3961 |

## Per-dataset test metrics

| Dataset | Queries | H@10 | R@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| clerk | 587 | 0.5332 | 0.3626 | 0.2936 | 0.3348 |
| cuad | 10455 | 0.0122 | 0.0122 | 0.0055 | 0.0035 |
| greek_bar_bench | 142 | 0.2746 | 0.2746 | 0.1282 | 0.0841 |
| lara | 1139 | 0.9350 | 0.9350 | 0.8432 | 0.8135 |

## Hypothesis verdicts

| Hypothesis | Verdict |
|---|---|
| H1 | NOT_SUPPORTED |
| H2 | PARTIALLY_SUPPORTED |
| H3 | SUPPORTED |
| H4 | TESTED_AS_EVALUATION_VIEW |
| H5 | SUPPORTED |
| H6 | NOT_TESTABLE |
| H7 | NOT_TESTABLE |

## Documented limitations

- Reduced dataset scope (4 of 7).
- Two controlled adaptations from M0 (numpy substrate, dense query re-exposure).
- H6, H7 not testable (no context builder, no reranker).
- Bootstrap SEs not multi-comparison corrected.
- Locked test dev->test delta: small, no overfitting evidence.
