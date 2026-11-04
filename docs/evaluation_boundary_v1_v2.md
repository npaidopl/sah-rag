# Evaluation Boundary: v1 (SAH-RAG historical) vs v2 (baselines)

**Status:** FROZEN as of 2026-11-04 (FIX-03 / F3)

## TL;DR

The repository contains **two independent evaluation stacks** that are
**NOT** mutually comparable in their raw metrics:

- **v1**: SAH-RAG historical evaluation (`scripts/run_sah_rag_m0_full_evaluation.py`)
- **v2**: Baseline evaluation (`evaluation/runner.py` + `evaluation/retrieval_metrics.py`)

**Do not** mix v1 and v2 numbers in the same comparison table.

## Why two stacks exist

SAH-RAG was evaluated (v1 paper, 2025) **before** the multi-baseline
comparison framework (v2, 2026) was designed. Rather than retroactively
modify the v1 script (which would invalidate historical numbers), the
project preserves both:

- **v1 script** = untouched historical implementation, produces v1 metrics
- **v2 runner** = new multi-baseline harness, produces v2 metrics
- **v2 boundary doc** = this file, explains the difference

## Concrete differences

| Aspect | v1 (SAH-RAG historical) | v2 (baseline harness) |
|---|---|---|
| **Cutoffs** | Inline `(1, 5, 10, 20, 50)` | `CUTOFFS = [1, 5, 10]` |
| **Precision K** | `PRECISION_K = 10` | `precision@10` only |
| **nDCG cutoffs** | `(10, 50)` | `{1, 5, 10}` |
| **hit@k** | Present (variable names match) | Present |
| **mrr@k** | Present | Present |
| **recall@k** | Present | Present |
| **precision@k** | Present | Present |
| **nDCG@k** | Present | Present |
| **Macro averaging** | Yes | Yes |
| **Micro averaging** | Yes | Yes |
| **Zero-relevance policy** | Excluded with report | Excluded with report |
| **metric_version field** | ABSENT | `"v2"` |
| **Artifact format** | `per_dataset_metrics.json` | `metrics.json` + `predictions.jsonl` |
| **Producer** | `scripts/run_sah_rag_m0_full_evaluation.py` | `evaluation/runner.py` |

## What this means

### You **CAN** compare within the same stack
- v1 number vs another v1 number
- v2 number vs another v2 number

### You **CANNOT** compare across stacks (without re-running)
- v1 SAH-RAG number **vs** v2 baseline number (in the same table)
- v1 SAH-RAG number **vs** v2 SAH-RAG number (they don't exist)

## How to get v2-compatible SAH-RAG numbers

**Option A (recommended):** Run SAH-RAG through the v2 runner.

    python -m orchestration.scheduler --manifest configs/sah_rag_v2_manifest.yaml

This produces SAH-RAG metrics using the same code path as baselines
(same cutoffs, same averaging, same zero-relevance policy), making
cross-baseline comparison valid.

**Option B:** Re-run the v1 script with v2 cutoffs (modify the script).

This is **discouraged** because:
- It changes historical reproducibility
- The v1 paper numbers use v1 cutoffs
- It duplicates infrastructure

## Historical v1 numbers (preserved for reference)

From the v1 paper (macro document Recall@10):

| Dataset | LongRAG | Metadata-RAG | SAH-RAG | Δ | % |
|---|---|---|---|---|---|
| CLERC | 0.0786 | 0.2726 | **0.3546** | +0.2760 | +351% |
| CUAD | 0.0324 | 0.0294 | **0.0431** | +0.0107 | +33% |
| GreekBarBench | 0.2606 | 0.2711 | **0.2852** | +0.0246 | +9.4% |
| LaRA | 0.8861 | 0.8874 | **0.9368** | +0.0507 | +5.7% |
| **Macro** | **0.3144** | **0.3651** | **0.4049** | +0.0905 | **+28.79%** |

**These are v1 numbers.** If you compare against v2 baseline numbers,
you MUST re-run SAH-RAG through v2 (Option A above).

## Long-term resolution

The project roadmap includes migrating SAH-RAG to the v2 runner for
all future comparisons. This is scheduled **after** the datasets
migration (FIX-07) and GPU execution phase. The v1 script will remain
in `scripts/` for historical reproducibility but will not be used for
new papers.

## References

- v1 script: `scripts/run_sah_rag_m0_full_evaluation.py`
- v2 runner: `evaluation/runner.py`
- v2 metrics: `evaluation/retrieval_metrics.py`
- v2 contracts: `evaluation/contracts.py`
- FROZEN boundary spec: `baseline_audit/evaluation_boundary.yaml`

## Change log

- **2026-11-04** (FIX-03 / F3): documented the two-stack difference.