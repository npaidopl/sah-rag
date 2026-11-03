# HiKEY Dataset Modality Audit

**Date:** 2026-11-04  
**Purpose:** Verify which datasets in the 17-dataset suite can be used for the native HiKEY pipeline (needs PDFs with tables/figures).  
**Compiled by:** automated audit script

---

## TL;DR

## Standard datasets: file stats

| Dataset | Raw (D:) | Raw (C: datasets) | Canonical output | PDFs found |
|---|---|---|---|---|
| 2wiki | 83.61 MB [.json×1, .parquet×1] | 54.95 MB [.json×1] | 69.69 MB [.json×2, .jsonl×2] | 0 |
| clerk | — | 37.83 MB [.ipynb×1, .json×3, .jsonl×3, .md×1, .py×1, .tag×1, .tsv×1, .txt×3, .yaml×1] | 42.59 MB [.json×3, .jsonl×3, .tsv×1] | 0 |
| cuad | — | 203.04 MB [(no ext)×2, .csv×4, .ipynb×1, .json×2, .md×1, .metadata×635, .pdf×502, .tag×1, .txt×200, .xlsx×29] | 85.59 MB [.json×2, .jsonl×2] | 511 |
| frames | 134.19 MB [.json×1, .jsonl×1, .tsv×1] | 66.96 MB [.json×1, .tsv×1] | 129.8 MB [.json×2, .jsonl×2] | 0 |
| greek_bar_bench | — | 128.15 MB [(no ext)×1, .arrow×1, .csv×1, .ipynb×1, .json×3, .jsonl×1, .log×1, .md×1, .parquet×2, .py×1, .tag×1, .yaml×1] | 74.1 MB [.json×2, .jsonl×2] | 0 |
| hotpotqa | 70.35 MB [.json×1, .parquet×1] | 44.17 MB [.json×1] | 91.46 MB [.json×2, .jsonl×2] | 0 |
| lara | — | 115.51 MB [(no ext)×12, .idx×1, .ipynb×1, .json×1, .jsonl×648, .md×51, .pack×1, .png×2, .py×12, .rev×1, .sample×14, .sh×2, .txt×82] | 81.13 MB [.json×3, .jsonl×2] | 0 |
| long_bench_v2 | — | 878.6 MB [.csv×1, .ipynb×1, .json×1] | 1295.6 MB [.json×3, .jsonl×2] | 0 |
| loong | — | 463.26 MB [.csv×1, .csv#×1, .ipynb×1, .parquet×1] | 952.34 MB [.json×3, .jsonl×2] | 0 |
| m3docvqa | 0.8 MB [.jsonl×1] | 0.8 MB [.jsonl×1] | 1.24 MB [.json×2, .jsonl×2] | 0 |
| multifieldqa_en | 353.99 MB [.jsonl×35] | 4.28 MB [.jsonl×1] | 12.84 MB [.json×2, .jsonl×2] | 0 |
| multihop_rag | 10.27 MB [.json×2] | 10.27 MB [.json×2] | 14.96 MB [.json×2, .jsonl×2] | 0 |
| musique | 29.03 MB [.jsonl×1] | 29.03 MB [.jsonl×1] | 27.28 MB [.json×2, .jsonl×2] | 0 |
| narrativeqa | 4670.69 MB [.jsonl×1] | 4670.69 MB [.jsonl×1] | 292.73 MB [.json×2, .jsonl×2] | 0 |
| qasper | 51.7 MB [.json×2, .md×1, .tgz×1] | 41.36 MB [.json×2] | 5.11 MB [.json×2, .jsonl×2] | 0 |
| quality | 18.77 MB [.jsonl×2] | 18.77 MB [.jsonl×2] | 16.61 MB [.json×2, .jsonl×2] | 0 |
| Ruler | — | 115.5 MB [.csv×1, .ipynb×1, .parquet×1] | 344.99 MB [.json×3, .jsonl×2] | 0 |

---

## M3DocVQA deep check

- Raw path exists: **True** (`D:\rag_work\raw_datasets\m3docvqa`)
- Canonical output exists: **True**
- Datasets dir exists: **True**
- Question count: **858**
- PDFs found (total): **0**
- Sample JSONL keys: `['question', 'answer', 'image_paths']`

---

## FRAMES deep check

- Raw path exists: **True** (`D:\rag_work\raw_datasets\frames`)
- Files:
  - `articles.jsonl` (67.22 MB)
  - `test.tsv` (0.46 MB)
  - `wiki_contexts.json` (66.5 MB)
- Articles count (articles.jsonl): **2521**
- Rendered PDFs available: **False** (0 PDFs)
- Playwright installed: **False**

---

## Classification: native HiKEY feasibility

| Dataset | Known type | PDFs | Native HiKEY? |
|---|---|---|---|
| 2wiki | TEXT_ONLY (Wikipedia) | 0 | ❌ NO (text-only) |
| clerk | TEXT_ONLY (legal) | 0 | ❌ NO (text-only) |
| cuad | TEXT_ONLY (contracts) | 511 | ⚠️ 511 PDFs found |
| frames | TEXT_ONLY (paper rendered to PDF) | 0 | ⚠️ TEXT-ONLY (renderable) |
| greek_bar_bench | TEXT_ONLY (legal) | 0 | ❌ NO (text-only) |
| hotpotqa | TEXT_ONLY (Wikipedia) | 0 | ❌ NO (text-only) |
| lara | TEXT_ONLY (long-context) | 0 | ❌ NO (text-only) |
| long_bench_v2 | TEXT_ONLY (long-context) | 0 | ❌ NO (text-only) |
| loong | TEXT_ONLY (multi-doc) | 0 | ❌ NO (text-only) |
| m3docvqa | PDF_BASED | 0 | ❌ NO PDFs |
| multifieldqa_en | TEXT_ONLY (long-context) | 0 | ❌ NO (text-only) |
| multihop_rag | TEXT_ONLY (news) | 0 | ❌ NO (text-only) |
| musique | TEXT_ONLY (Wikipedia) | 0 | ❌ NO (text-only) |
| narrativeqa | TEXT_ONLY (books + scripts) | 0 | ❌ NO (text-only) |
| qasper | TEXT_ONLY (scientific papers as text) | 0 | ❌ NO (text-only) |
| quality | TEXT_ONLY (articles) | 0 | ❌ NO (text-only) |
| Ruler | TEXT_ONLY (synthetic) | 0 | ❌ NO (text-only) |

---

## Decision

### Controlled track (text-only approximation)

**Status: TOTALLY_READY** for all 17 datasets. The controlled adapter consumes `CanonicalDocument`, which is the canonical format that all 17 datasets already provide.

### Native track (multimodal, PDFs + tables/figures)

**M3DocVQA**: 0 PDFs available. This enables a **smoke-test** native run (singlehop subset). Not enough for full evaluation (paper uses 3,300 PDFs).

**FRAMES-PDF**: 2521 articles available in `articles.jsonl`, but PDFs not yet rendered. To match the paper, render with Playwright (the paper used Chromium + Playwright).

### Blockers (still external)

1. **GPU access**: gte-Qwen2-7B + SigLIP need ~15 GB VRAM.
2. **Full M3DocVQA**: 3,300 PDFs require Kaggle account.
3. **FRAMES-PDF rendering**: run Playwright script (1-2 hours for 2,476 articles).