# LongRAG Full-Track Closure Report

**Fidelity class:** CONTROLLED_ADAPTATION

## Scope

Datasets executed: clerk, cuad, greek_bar_bench, lara

## Per-dataset summary

| Dataset | Units | Queries | H@10 | R@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| clerk | 256 | 1000 | 0.0940 | 0.0786 | 0.0543 |
| cuad | 1725 | 20910 | 0.0324 | 0.0324 | 0.0153 |
| greek_bar_bench | 1437 | 284 | 0.2606 | 0.2606 | 0.1233 |
| lara | 2251 | 2326 | 0.8861 | 0.8861 | 0.7977 |
| **MACRO** | | | 0.3183 | 0.3144 | 0.2476 |

## Primary metric

`macro_document_Recall_at_10 = 0.314405`

## Documented limitations

- Reduced dataset scope: 4 of 7 canonical datasets (ruler, loong, long_bench_v2 excluded by user decision).
- Unit granularity on small-document corpora (clerk, cuad, greek_bar_bench) causes coarse document-level ranking. This is a finding about LongRAG's design when the corpus lacks Wikipedia-style hyperlink adjacency.
- Document expansion uses first_occurrence_by_long_unit_rank, matching the reference implementation.
- FP16 embeddings for documents and queries using the pinned BGE-M3 revision.
- cuad: 14208/20910 queries have zero relevant documents and are excluded from metrics.
