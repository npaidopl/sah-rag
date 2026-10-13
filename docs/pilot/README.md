# Infrastructure Pilot Documentation

Design and execution documentation for the infrastructure pilot. The pilot
preceded the full-track evaluation and was used to validate the end-to-end
pipeline on a small synthetic dataset before scaling to the real corpora.

## Design

| File | Stage |
|---|---|
| `pilot_parser_chunker_execution.md` | Parser + chunker pilot execution |
| `pilot_embedding_execution_plan.md` | Embedding pilot plan |
| `pilot_embedding_benchmark.md` | Embedding benchmark results |
| `pilot_qdrant_ingestion_plan.md` | Qdrant ingestion plan |
| `pilot_qdrant_bounded_ingestion.md` | Bounded ingestion execution |
| `pilot_hybrid_retrieval_plan.md` | Hybrid retrieval plan |
| `pilot_hybrid_retrieval_execution.md` | Hybrid retrieval execution |
| `pilot_retrieval_evaluation_plan.md` | Evaluation plan |
| `pilot_retrieval_evaluation_execution.md` | Evaluation execution |
| `pilot_canonical_adapter.md` | Canonical adapter for pilot data |

## Relation to full-track evaluation

The pilot used a synthetic infrastructure dataset. It is documented here
for completeness; the primary results come from the full-track evaluation
on the four canonical datasets (CLERC, CUAD, GreekBarBench, LaRA).

See [thesis `docs/data/`](../data/) and [reports `docs/reports/`](../reports/)
for the full-track results.
