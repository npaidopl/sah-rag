# HiKEY (Reported Baseline)

**Paper:** Shin et al., "HiKEY: Hierarchical Multimodal Retrieval for Open-Domain Document Question Answering", arXiv:2605.29606v1, ACL 2026.

**Status:** REPORTED. No official code exists.

## Why reported-only

There is no public repository, no author-released configs, and no model weights for HiKEY. A native executable baseline cannot be built honestly. Native execution would require reimplementation with guessed details.

## What this adapter exposes

- `get_reported_metrics(dataset_id)` returns the metric dict from `reported_metrics.yaml`.
- Only `M3DocVQA`, `FRAMES-PDF`, `AVG` are available. Any other dataset returns an empty dict.

## What this adapter does NOT do

- It does not run any code.
- It does not implement HiKEY.
- It cannot be used to claim SAH-RAG beats HiKEY on datasets other than M3DocVQA/FRAMES-PDF.

## Comparison scope rule

Any "SAH-RAG vs HiKEY" claim MUST be restricted to M3DocVQA or FRAMES-PDF, and MUST use paper-reported numbers (verbatim). Comparisons on other datasets are invalid.

## Related artifacts in this project

- `workspace/hikey_native_dev/` - paper-derived native pipeline (standalone, IR-focused).
- `workspace/hikey_dev/` - text-only controlled approximation (clearly labeled, not HiKEY).
- `docs/hikey_ablation_proposal.md` - proposed SAH-RAG ablation components inspired by HiKEY.