# FIX-05 Verification: Multi-Meta-RAG Controlled Mode

**Status:** VERIFIED (2026-11-04)

## Summary

Multi-Meta-RAG controlled mode executed end-to-end on the pilot dataset
`infrastructure-pilot-synthetic-v1` (12 docs, 6 queries).

## Test results

### Adapter tests (11/11 passed)

- test_controlled_build_creates_index_files
- test_controlled_retrieve_contiguous_ranks
- test_controlled_respects_top_k
- test_controlled_filter_by_source
- test_controlled_no_filter_when_no_source_in_query
- test_controlled_missing_source_metadata
- test_native_requires_prereqs
- test_native_llm_extraction_filters
- test_native_null_source_returns_all
- test_native_chunking_produces_multiple_chunks
- test_native_and_controlled_same_interface

### Controlled E2E run

**Manifest:** `configs/pilot_multi_meta_rag_controlled.yaml`

**Execution:**

    python -m orchestration.scheduler configs/pilot_multi_meta_rag_controlled.yaml

**Result:** Exit code 0, DONE in ~32 seconds.

**Metrics (metric_version=v2, cutoffs=[1,5,10]):**

| Metric | @1 | @5 | @10 |
|---|---|---|---|
| recall | 0.5 | 1.0 | 1.0 |
| mrr | 1.0 | 1.0 | 1.0 |
| ndcg | 1.0 | 1.0 | 1.0 |
| hit | 1.0 | 1.0 | 1.0 |

**Predictions sample (q1):**

- GT: pilot-doc-01, pilot-doc-02
- Retrieved: pilot-doc-01 (rank 1, score 0.749), pilot-doc-02 (rank 2, score 0.749),
  pilot-doc-05 (rank 3, score 0.505), ...
- Correct ordering: both GT docs ranked in top-2 with strong score separation.

## Key insight

Multi-Meta-RAG controlled mode is **LLM-free**:
- `is_llm_based = False`
- Controlled mode uses deterministic substring matcher for metadata extraction
- Injects BGE-M3 embedder (dense-only) but **not** Qwen 14B
- **CPU-feasible** for small corpora

This means FIX-05 requires **no GPU**, **no API key**, and **no Qwen weights**.
The run is fully reproducible on CPU in <1 minute.

## Files

- Manifest: `configs/pilot_multi_meta_rag_controlled.yaml`
- Adapter: `baselines/multi_meta_rag/adapter.py`
- Tests: `baselines/multi_meta_rag/tests/test_adapter.py`
- Runner dispatch: `evaluation/runner.py::_build_controlled_components`

## What this confirms

For the run_all.py orchestrator, Multi-Meta-RAG controlled cells will:
- Not require `RAG_INDOOR_LLM_ROOT` env var
- Not require GPU
- Execute in ~30s per (baseline × dataset) cell
- Produce standard `metrics.json` + `predictions.jsonl` artifacts

## Change log

- 2026-11-04: verified end-to-end (FIX-05).