# run_all.py Usage Guide

**Last updated:** 2026-11-04

Complete reference for executing the SAH-RAG research pipeline
(`orchestration/run_all.py`) on the university GPU server.

## Overview

`run_all.py` is the master orchestrator that executes four research
pipelines sequentially:

| Stage | Name | What it does | Time (GPU) |
|---|---|---|---|
| 0 | Preflight | Check env, models, datasets, workspace | < 1 min |
| 1 | Optimal Operation | M2-M8 pipeline (subprocess) | 3-6 hours |
| 2 | Ablation Study | 26 variants across tiers (subprocess) | 3-6 hours |
| 3 | Controlled Adaptation | 10 baselines x 17 datasets = 170 cells | 24-72 hours |
| 4 | Native E2E | 10 baselines x 17 datasets = 170 cells | 12-48 hours |
| 5 | Reporting | Aggregate metrics + closure manifest | < 30 min |

**Total (worst case):** ~5 days continuous GPU time.

## Pre-requisites

### 1. Environment variables

Must be set in `.env` (in project root):

```bash
RAG_CANONICAL_ROOT=C:/path/to/data/processed/canonical
RAG_INDOOR_LLM_ROOT=D:/rag_work/models/Qwen2.5-14B-Instruct
RAG_GENERATION_LLM_ROOT=D:/rag_work/models/Qwen2.5-32B-Instruct
RAG_EMBEDDER_ROOT=D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/<rev>
```

Optional (native E2E baselines will fail per-cell if missing, but the
overall run continues):

```bash
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

Optional (enables Stage 1 Optimal Operation):

```bash
SAH_RAG_OPTIMAL_WORKSPACE=D:/rag_work/sah_rag_optimal_operation
```

### 2. Models on disk

| Model | Size | Required for |
|---|---|---|
| Qwen2.5-14B-Instruct | ~28 GB | Controlled indoor LLM |
| Qwen2.5-32B-Instruct | ~64 GB | Controlled generation (optional) |
| BGE-M3 | ~4.3 GB | Embeddings |
| BGE-large-en-v1.5 | ~2.5 GB | Native (Multi-Meta-RAG, LongRAG) |
| naver/splade-v3 | ~440 MB | SPLADE native |
| colbert-ir/colbertv2.0 | ~440 MB | ColBERTv2 native |
| Qwen2.5-1.5B-Instruct | ~2.9 GB | RAPTOR native, LightRAG native |
| Qwen2.5-7B-Instruct | ~14 GB | MemoRAG native |

### 3. Datasets

17 canonical datasets must be in `RAG_CANONICAL_ROOT`:

```text
2wiki, clerk, cuad, frames, greek_bar_bench, hotpotqa, lara,
long_bench_v2, loong, m3docvqa, multifieldqa_en, multihop_rag,
musique, narrativeqa, qasper, quality, ruler
```

Verify with:

```bash
python -m orchestration.run_all --manifest configs/run_manifest_full.yaml --stage 0
```

## How to run

### Quick start

```bash
cd /path/to/SAH-RAG
source .venv/bin/activate
python -m orchestration.run_all --manifest configs/run_manifest_full.yaml
```

### Dry-run (planning only, no execution)

```bash
python -m orchestration.run_all --manifest configs/run_manifest_full.yaml --dry-run
```

> **Note:** `--dry-run` only affects stages 3 and 4 (they print the cell list
> without running). Stages 1 and 2 still execute their subprocesses. To
> verify only preflight: `--stage 0`.

### Resume after interruption

```bash
python -m orchestration.run_all --manifest configs/run_manifest_full.yaml --resume
```

The state is persisted to `outputs/run_all/run_all_state.json` after
every cell. If the run is interrupted, `--resume` skips completed cells.

### Run a single stage

```bash
# Only preflight
python -m orchestration.run_all --manifest configs/run_manifest_full.yaml --stage 0

# Only controlled adaptation (if you want to skip native)
python -m orchestration.run_all --manifest configs/run_manifest_full.yaml --stage 3
```

### Force re-run a stage

```bash
python -m orchestration.run_all --manifest configs/run_manifest_full.yaml --stage 3 --force-stage
```

## What gets produced

All outputs go to `outputs/run_all/` (configurable via `global.output_root`):

```text
outputs/run_all/
├── run_all_state.json                  # Stage-level state (resume info)
├── _generated_controlled_adaptation.yaml  # On-the-fly manifest
├── _generated_native_e2e.yaml
├── controlled_adaptation/
│   ├── index.db                        # SQLite run index
│   ├── controlled__bm25__clerk__<hash>/metrics.json
│   ├── controlled__bm25__clerk__<hash>/predictions.jsonl
│   ├── controlled__bm25__clerk__<hash>/DONE  (marker)
│   └── ... (170 cells)
├── native_e2e/
│   └── ... (170 cells)
└── reports/
    ├── summary.json                    # Aggregate metrics
    └── closure_manifest.json           # Full provenance
```

## How to monitor progress

### Live monitoring

```bash
# Watch the run_all.log (if redirecting output)
tail -f run_all.log

# Or check state.json every few minutes
watch -n 60 "cat outputs/run_all/run_all_state.json | python -m json.tool | head -50"
```

### Check cell progress

```bash
# Count done cells in controlled_adaptation
find outputs/run_all/controlled_adaptation -name "DONE" | wc -l

# Count failed cells
find outputs/run_all/controlled_adaptation -name "error.log" | wc -l
```

### SQLite index

```bash
sqlite3 outputs/run_all/controlled_adaptation/index.db \
    "SELECT status, COUNT(*) FROM runs GROUP BY status"
```

Expected output:

```text
done|170
failed|0
```

## How to resume a stopped run

The most common scenarios:

### A. SSH disconnect

```bash
# Reconnect
ssh user@gpu
cd /path/to/SAH-RAG
source .venv/bin/activate
python -m orchestration.run_all --manifest configs/run_manifest_full.yaml --resume
```

### B. GPU out of memory mid-run

Check `outputs/run_all/controlled_adaptation/*/error.log` for OOM errors.
Fix the model loading (e.g., switch to 14B generation), then resume.

### C. Model not found

Preflight would have caught it. If it slips through (e.g., API key), the
affected cells fail with a clear error but the run continues (`allow_partial=true`).

## Expected exit codes

| Exit code | Meaning |
|---|---|
| 0 | All stages done, no failures |
| 1 | Manifest not found |
| 2 | Stage failed fatally (`stop_on_fatal=true`) or preflight failed |

## Configuring the run

Edit `configs/run_manifest_full.yaml`:

### Disable a stage

```yaml
optimal_operation:
  enabled: false    # skip Stage 1
```

### Run only 5 datasets (faster, for testing)

```yaml
controlled_adaptation:
  datasets:
    - clerk
    - cuad
    - qasper
    - hotpotqa
    - lara
```

### Run only controlled (no native)

```yaml
native_e2e:
  enabled: false
```

### Change generation LLM to 14B

```bash
# In .env
# RAG_GENERATION_LLM_ROOT=D:/rag_work/models/Qwen2.5-14B-Instruct
```

## Troubleshooting

### "RAG_CANONICAL_ROOT is not configured"

`.env` not found or not loaded. Verify:

```bash
ls -la .env
cat .env | grep RAG_CANONICAL_ROOT
```

Then re-run. `run_all.py` loads `.env` at startup automatically.

### "QwenLLM requires explicit model_path"

`RAG_INDOOR_LLM_ROOT` not set. Add to `.env`.

### "RAG_INDOOR_LLM_ROOT is not set. Controlled mode requires Qwen2.5-14B-Instruct"

Same as above.

### Preflight passes but cells fail

Look at `outputs/run_all/*/index.db` for failed cells:

```bash
sqlite3 outputs/run_all/controlled_adaptation/index.db \
    "SELECT run_id, baseline, dataset, error FROM runs WHERE status='failed' LIMIT 10"
```

Then check the corresponding `error.log`:

```bash
cat outputs/run_all/controlled_adaptation/controlled__bm25__clerk__<hash>/error.log
```

### Model load OOM

If GPU has <40GB VRAM:

- 14B indoor (~28GB) fits
- 32B generation (~64GB) doesn't fit
- Use `--stage 3` with `RAG_GENERATION_LLM_ROOT=RAG_INDOOR_LLM_ROOT` (14B fallback)

### Datasets missing

Run `python -m orchestration.run_all --stage 0` to see exactly which
dataset is missing, then verify it exists at:

```bash
ls $RAG_CANONICAL_ROOT/<dataset>/
```

Should contain: `documents.jsonl`, `examples.jsonl`, `validation.json`, `manifest.json`

## Expected outputs per cell

Each successful cell writes:

### metrics.json

```json
{
  "run_id": "abc123def4567890",
  "baseline": "bm25",
  "dataset": "clerk",
  "mode": "controlled",
  "metric_version": "v2",
  "cutoffs": [1, 5, 10],
  "top_k": 10,
  "n_documents": 2756,
  "n_examples": 1000,
  "n_evaluated": 1000,
  "started_at": "...",
  "finished_at": "...",
  "metrics": {
    "recall@1": 0.xx,
    "recall@5": 0.xx,
    "recall@10": 0.xx,
    "mrr@10": 0.xx,
    "ndcg@10": 0.xx,
    "hit@10": 0.xx
  }
}
```

### predictions.jsonl

One JSON object per query with retrieved doc IDs, ranks, scores, and
ground truth for full reproducibility.

### DONE

Marker file (empty). Indicates the cell completed successfully.

## Interpreting Stage 5 output

`outputs/run_all/reports/summary.json`:

```json
{
  "generated_at": "...",
  "total_runs": 340,
  "runs_by_baseline": {
    "bm25": 34,     // 17 controlled + 17 native
    "splade": 34,
    ...
  },
  "runs_by_dataset": {
    "clerk": 20,    // 10 controlled + 10 native
    ...
  },
  "runs_by_mode": {
    "controlled": 170,
    "native": 170
  }
}
```

`outputs/run_all/reports/closure_manifest.json`: full state + manifest + all metrics paths.

## Post-run: what to do

1. **Verify all cells completed:**

   ```bash
   sqlite3 outputs/run_all/*/index.db "SELECT status, COUNT(*) FROM runs GROUP BY status"
   ```

   Target: `done=340, failed=0`

2. **Copy reports to your analysis environment:**

   ```bash
   scp -r outputs/run_all/reports user@workstation:~/analysis/
   ```

3. **Analyze:**
   - Per-baseline comparison: use `reports/summary.json`
   - Per-dataset breakdown: query SQLite or parse `metrics.json` files
   - Statistical tests: apply paired bootstrap on `predictions.jsonl`

4. **Publish:**
   - Reports are in `outputs/run_all/reports/`
   - Raw data in `outputs/run_all/{controlled,native}_e2e/*/`
   - Everything is SHA-256 traceable via `DONE` markers and closure manifest

## Known limitations

- **No streaming generation:** Cells run to completion before returning metrics.
- **Single GPU:** No parallelism. Multi-GPU support is deferred.
- **No automatic retry:** Failed cells stay failed; use `--force-stage` to retry.
- **CPU-only run:** Stages 3 and 4 require GPU. Stage 0 and 5 work on CPU. Stage 1/2 partial (skip-gpu).

## Change log

- **2026-11-04:** Initial guide. Corresponds to commit `71159c6`.
