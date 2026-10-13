# Documentation

This directory contains the written documentation of the SAH-RAG project.

## Contents

### Reports (`reports/`)

LaTeX sources of all written reports. See [`reports/README.md`](reports/README.md).

- `sah_rag_implementation.tex` — SAH-RAG architecture & implementation
- `sah_rag_optimization.tex` — SAH-RAG parameter study
- `datasets.tex` — Dataset descriptions
- `pipelines_status.tex` — Pipelines status
- `longrag_full_track_report.tex` — LongRAG comparator
- `longrag_vs_sah_rag.tex` — LongRAG vs SAH-RAG
- `metadata_rag_full_track_report.tex` — Metadata-RAG comparator
- `three_way_comparison.tex` — SAH-RAG vs LongRAG vs Metadata-RAG

### Project (`data/`)

- `REPORT_NIKOLAOS_PAIDOPOULOS.pdf` — Full Diploma Project

### Pilot design docs (`pilot/`)

Infrastructure pilot documentation. See [`pilot/README.md`](pilot/README.md).

### Top-level documentation

| File | Purpose |
|---|---|
| `architecture_specification.md` | SAH-RAG architecture, components, invariants |
| `fair_benchmark_methodology.md` | Shared fair-comparison protocol |
| `canonical_data_contract.md` | Canonical envelope and data model |
| `deterministic_parser.md` | Structural parser specification |
| `hierarchical_child_chunker.md` | Child chunker specification |
| `hybrid_retrieval.md` | Hybrid RRF retrieval and aggregation |
| `qdrant_hybrid_storage.md` | Qdrant collection layout |
| `embedding_shards.md` | Embedding shard format |
| `bge_m3_tokenizer_adapter.md` | Tokenizer adapter details |
| `artifact_provenance.md` | Content-addressable identities and manifests |
| `execution_environment_freeze.md` | Environment reproduction policy |
| `docker_qdrant_operations.md` | Local Qdrant operations |
| `external_canonical_dataset_suite.md` | External dataset suite |
| `real_dataset_inventory_and_sizing.md` | Dataset inventory and sizing |
| `retrieval_evaluation.md` | Evaluation procedure |
| `metadata_rag_paper_spec_audit.md` | Metadata-RAG clean-room spec audit |
| `longrag_full_track_closure.md` | LongRAG full-track closure |
| `longrag_vs_sah_rag_comparison.md` | LongRAG vs SAH-RAG comparison write-up |
| `sah_rag_m0_closure.md` | SAH-RAG M0 closure |
| `sah_rag_m7_locked_test.md` | SAH-RAG M7 locked test |
| `sah_rag_m8_closure.md` | SAH-RAG M8 closure |
| `sah_rag_end_to_end_inspection.md` | End-to-end inspection |
| `remaining_baselines_sequencing.md` | Disposition of non-executed comparators |

## Frozen contracts

Frozen experimental contracts live in the sibling directory
[`baseline_audit/`](../baseline_audit/). These files are read by validators
and are not modified after the corresponding experiment has been committed.

## Reproducibility

Every run produces a `manifest.json` with SHA-256 checksums of all inputs and
outputs. To verify a run:

    python scripts/validate_*.py --manifest path/to/manifest.json

See [`scripts/`](../scripts/) for details.
