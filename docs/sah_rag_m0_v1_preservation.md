# SAH-RAG v1 Historical Artifacts Preservation

**Date:** 2026-11-04
**Status:** PRESERVED
**Purpose:** Document the migration of v1 SAH-RAG artifacts to the current project.

## Overview

The v1 SAH-RAG was developed in the legacy project at
`C:\Users\USER1\Documents\RAG Architecture` under the code name **my_rag**
(later renamed to **sah_rag**). The v1 metrics reported in the thesis PDF
(macro Recall@10 = 0.4049, +28.79% vs LongRAG, +351% on CLERC) were produced
by that version and stored in `outputs/external_my_rag/`.

For reproducibility of the paper's historical claims, those artifacts have been
copied to `outputs/external_sah_rag/` in the current project.

## Source and target

- **Source (v1):** `C:\Users\USER1\Documents\RAG Architecture\outputs\external_my_rag\`
- **Target (v2):** `outputs/external_sah_rag/` (this repo)
- **Size:** 39.4 GB
- **Copy date:** 2026-11-04
- **Copy method:** robocopy /E /MT:8
- **SHA-256 verified:** 4/4 per_dataset_metrics.json match

## Structure
outputs/external_sah_rag/
├── clerk/
│ ├── chunks-v1/ (chunks + manifest)
│ ├── embeddings-v1/ (embedding_records.jsonl)
│ ├── evaluation-v1/ (per_dataset_metrics.json)
│ ├── retrieval-v1/ (results.jsonl)
│ ├── grid-chunks-v1/ (8 config grid: c256/c512/c1024 × o0/o32/o128)
│ ├── grid-embeddings-v1/ (8 config grid embeddings)
│ └── parsed-v1/ (semantic parents)
├── cuad/ (same structure)
├── greek_bar_bench/ (same structure)
├── lara/ (same structure)
├── _m0_cross_dataset_evaluation.json (MACRO numbers)
├── _m0_l2_root_manifest.json
├── _m0_l5_root_manifest.json
├── _colab_v1/ (Colab embedding bundles)
├── _colab_queries_v1/ (Colab query bundles)
├── _grid_colab_v1/ (grid Colab bundles)
├── _qdrant/ (Qdrant storage manifest)
├── _query_embeddings/ (per-dataset query embeddings)
├── _m2_grid_root_manifest.json (grid search manifest)
├── _m4a_screening_manifest.json (parameter screening)
├── _m4a_screening_results.jsonl
├── m4b.json (facet analysis)
├── m5.jsonl (confirmatory)
├── m6.json (selection)
├── m7_locked_test.jsonl (locked test)
└── _optimization_split.* (split manifest)
## v1 numbers (verified from _m0_cross_dataset_evaluation.json)

| Dataset        | Hit@10 | Recall@10 | nDCG@10 |
|----------------|--------|-----------|---------|
| CLERC          | 0.5430 | 0.3546    | 0.2875  |
| CUAD           | 0.0431 | 0.0431    | 0.0199  |
| GreekBarBench  | 0.2852 | 0.2852    | 0.1314  |
| LaRA           | 0.9368 | 0.9368    | 0.8466  |
| **MACRO**      | 0.4520 | **0.4049** | 0.3214 |

**Primary metric:** `macro_document_Recall_at_10 = 0.4049358066559581`

## IMPORTANT: v1 vs v2 evaluation boundary

These numbers are **v1 semantics**:
- CUTOFFS = (1, 5, 10, 20, 50)
- No `metric_version` field
- Artifact type: `my_rag_m0_evaluation_report`

They are **NOT directly comparable** to v2 baseline numbers (which use
CUTOFFS = [1, 5, 10], `metric_version = "v2"`, `hit@k` field).

For v2-compatible SAH-RAG numbers, the v1 artifacts must be re-evaluated
through the v2 evaluator (see `docs/evaluation_boundary_v1_v2.md`).

## Usage

**For the paper's historical claims:** cite the v1 numbers directly with
the explicit disclaimer "v1 metric semantics, not directly comparable to v2
baselines".

**For v2-compatible comparison:** re-run the v1 retrieval results through
`evaluation/retrieval_metrics.py` with v2 cutoffs. This requires the
`retrieval-v1/results.jsonl` files (preserved here) plus the v2 evaluator.

## gitignore

This folder is **gitignored** (39.4 GB is too large for git). It lives only
on the local filesystem. To reproduce from scratch:

1. Migrate the raw chunks from the legacy project
2. Re-run the v1 pipeline (`scripts/run_sah_rag_m0_*.py`)
3. OR restore from a backup if available

## Preservation checklist

- [x] Copied source → target
- [x] Size match: 39.4 GB
- [x] SHA-256 verification: 4/4 per_dataset_metrics.json
- [x] Macro numbers confirmed: 0.4049
- [x] Documentation created

## Change log

- **2026-11-04:** Initial preservation. Migrated from legacy project.