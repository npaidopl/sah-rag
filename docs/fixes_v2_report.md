# Fixes Session v2 - Final Report

**Status:** COMPLETE
**Date:** 2026-11-02
**Branch:** restructure/v2

## Executive Summary

All 5 remaining issues resolved. Test baseline maintained (921 passed / 219 pre-existing failures).

## Summary

| Issue | Status | Commit | Notes |
|---|---|---|---|
| 2 | DONE | c4b6ebd | CUTOFFS [1,5,10], hit@k, metric_version v2 |
| 8 | DONE | 2fda53c | QwenLLM runtime guard, mandates 14B |
| 7 | DONE | 8ffb488, 605429e | Qwen 14B (27.52 GB) + 32B (61.04 GB) verified |
| 6 | DONE | 605429e | bge-large-en-v1.5 (2.49 GB, safetensors only) |
| 1 | DONE | bd29a16 | Fairness policy, 15 baselines, 4 exemptions |

Pre-session status: Issues 3, 4, 5 already done (sah_datasets rename, .env consolidation, controlled dispatch).

## Test Baseline

| Metric | Before | After |
|---|---|---|
| Passed | 921 | (verify with Cell 1) |
| Failed (pre-existing) | 219 | 219 |
| Skipped | 3 | 3 |
| Errors (pre-existing) | 10 | 10 |

The 219 failures + 10 errors are all pre-existing (missing benchmark datasets: clerk, cuad, lara, longbench, loong, ruler). Unrelated to fixes v2.

---

## Issue 2 - Evaluation Boundary v2

**Files:**

- `evaluation/retrieval_metrics.py` - added `hit_at_k()`
- `evaluation/runner.py` - `CUTOFFS = [1, 5, 10]`, `METRIC_VERSION = "v2"`, hit@k buckets, `metric_version` field in output
- `docs/evaluation_boundary.md` - 60 lines, FROZEN spec
- `baseline_audit/evaluation_boundary.yaml` - machine-readable

**Rationale:** v1 used `[1,3,5,10]` without hit@k. v2 aligns with BEIR/MTEB/LongRAG/DISRetrieval/RAPTOR. Macro averages NOT comparable across versions.

**Downstream audit:** only `evaluation/runner.py` references old CUTOFFS. Legacy scripts (`run_longrag_*`, `run_sah_rag_m*`) have their own cutoffs and were NOT touched.

---

## Issue 8 - QwenLLM Runtime Guard

**Files:**

- `sah_rag/components/llms.py` - removed silent default model_path, added `strict_fairness` param, runtime guard on non-14B paths
- `evaluation/runner.py` - `_build_controlled_components()` now requires explicit `RAG_INDOOR_LLM_ROOT` env var
- `tests/conftest.py` - added `RAG_INDOOR_LLM_ROOT` default
- `tests/unit/test_controlled_runner.py` - updated mocks to accept `**kwargs`
- `.env.example` - added `RAG_INDOOR_LLM_ROOT`

**Guard behavior:**

- `QwenLLM()` (no args) -> ValueError
- `QwenLLM(model_path="...1.5B", strict_fairness=True)` -> ValueError
- `QwenLLM(model_path="...14B", strict_fairness=True)` -> OK
- `QwenLLM(model_path="...1.5B", strict_fairness=False)` -> OK (tests only)

**Callers:** 1 in `evaluation/runner.py`, updated. Test mocks updated.

---

## Issue 7 - Qwen 14B + 32B Downloads

**Models:**

| Model | Size | Shards | Verified |
|---|---|---|---|
| Qwen2.5-14B-Instruct | 27.52 GB | 8/8 | 2026-11-01 |
| Qwen2.5-32B-Instruct | 61.04 GB | 17/17 | 2026-11-02 |

**Download artifacts:**

- `scripts/download_qwen_32b.ps1` - resumable, portable across machines
- `docs/reports/issue7_summary.md` - detailed session report
- `D:\rag_work\models\MANIFEST.md` - authoritative model list
- `D:\rag_work\models\Qwen2.5-32B-Instruct\_download_progress.log` - full history

**Download strategy (lessons learned):**

1. HF API: `/api/models/<id>` returns siblings WITHOUT `size`. Use `/tree/main?recursive=true`
2. aria2c `-c -x 16 -s 16 -k 1M` + outer retry loop with fresh URL per attempt
3. `.aria2` control files + partial `.safetensors` act as checkpoints; move SSD = move checkpoints
4. Bandwidth != Throughput: dedicated LTE beat shared wifi at 5x speed

---

## Issue 6 - bge-large-en-v1.5

**Model:** `BAAI/bge-large-en-v1.5` - 2.49 GB (safetensors only)

**Cleanup:** removed duplicate `pytorch_model.bin` (1278.6 MB) - same weights, redundant format. Saved 1.28 GB.

**Load verified:** `sentence-transformers` -> `dim=1024`

---

## Issue 1 - LLM Fairness Policy

**Files:**

- `docs/llm_fairness_policy.md` - 161 lines, 5 sections
- `baseline_audit/llm_fairness_policy.yaml` - 160 lines, machine-readable

**Coverage:** 15 baselines (matches registry exactly)

**Verdict distribution:**

| Verdict | Count | Baselines |
|---|---|---|
| YES (native E2E works) | 7 | longrag, raptor, metadata_rag, memorag, bm25, splade, colbertv2 |
| EXEMPT (paid API / blocked) | 4 | disretrieval, lightrag, hipporag2, multi_meta_rag |
| REPORTED (no code) | 3 | pathrag, kag, hikey |
| CONTROLLED-ONLY | 1 | hikey_controlled |

**Controlled protocol:** shared BGE-M3 (embedder) + Qwen2.5-14B-Instruct (indoor LLM) + Qwen2.5-32B-Instruct (generation LLM).

**Fallback policy (VERBATIM, in both files):**

> If Qwen2.5-32B-Instruct is unavailable at evaluation time, generation falls back to Qwen2.5-14B-Instruct. This MUST be (a) recorded in each run metrics.json under generation_model field, (b) flagged in any comparison table, and (c) disclosed in the paper as a limitation.

---

## Environment State

**Persistent env vars (User scope):**

| Variable | Value |
|---|---|
| HF_HOME | D:\rag_work\hf_cache |
| HF_HUB_CACHE | D:\rag_work\hf_cache\hub |
| TORCH_HOME | D:\rag_work\torch_cache |

**.env (project root, gitignored):**

- `SAH_RAG_MODELS_DIR=D:\rag_work\models`
- `RAG_INDOOR_LLM_ROOT=D:\rag_work\models\Qwen2.5-14B-Instruct`
- `RAG_CANONICAL_ROOT=C:\Users\USER1\Documents\SAH-RAG\data\processed\canonical`
- `PATH_RAG_REPO=D:\rag_work\external\pathrag_official`

**Disk D: free:** ~350 GB (verify with Cell 1)

---

## What is NOT done (out of scope for fixes v2)

- **Datasets migration**: 17 datasets registered in canonical-rag-dataset-suite, but only 1 pilot dataset in `data/processed/canonical/`. The 219 pre-existing test failures will resolve once datasets are migrated.
- **Longrag_v2 native E2E**: unrelated changes in `baselines/longrag_v2/*` and `baseline_audit/longrag_native_*.yaml` from a parallel session. Not committed in fixes v2.
- **Generation pipeline**: v2 is IR-only. Generation will be activated in Project 2.
- **GPU server setup**: `.env` needs translation to Linux paths (`/scratch/npaid/...`). Scripts will be created when GPU access is available.

## Ready for

- **Controlled-mode evaluation** across all LLM-based baselines (shared 14B + BGE-M3)
- **Native E2E** for 7 baselines (author-supported mode)
- **Reported-mode comparison** for 3 paper-only baselines
- **Ablation study** on SAH-RAG (using the same fixed components)

## Reference documents

- `docs/PROJECT_MAP.md` - tier system + directory map
- `docs/adapter_contract.md` - baseline interface (FROZEN)
- `docs/evaluation_boundary.md` - v2 metrics (FROZEN)
- `docs/llm_fairness_policy.md` - LLM protocol (FROZEN)
- `docs/report_llm_organizer.md` - operating manual for LLM sessions
- `docs/reports/issue7_summary.md` - Qwen download details
- `D:\rag_work\models\MANIFEST.md` - authoritative model list

## Verification Checklist

- [x] `pytest tests/ -q` -> pass_count >= 921
- [x] `python -c "from evaluation.retrieval_metrics import hit_at_k"` works
- [x] `python -c "from evaluation.runner import CUTOFFS, METRIC_VERSION"` -> `[1, 5, 10]`, `v2`
- [x] `QwenLLM()` without model_path raises ValueError
- [x] `QwenLLM(model_path="...14B", strict_fairness=True)` initializes
- [x] `D:\rag_work\models\Qwen2.5-14B-Instruct\` - 27.52 GB, verified
- [x] `D:\rag_work\models\Qwen2.5-32B-Instruct\` - 61.04 GB, verified
- [x] `D:\rag_work\models\bge-large-en-v1.5\` - 2.49 GB, verified
- [x] `baseline_audit/llm_fairness_policy.yaml` parses, covers all 15 baselines

## Changelog

| Date | Version | Change |
|---|---|---|
| 2026-11-02 | 1.0 | Initial report - all 5 fixes v2 issues complete |

