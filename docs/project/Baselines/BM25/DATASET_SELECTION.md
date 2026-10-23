# BM25 Dataset Selection β€” Recommendation

> **Status:** awaiting supervisor decision.
> **Purpose:** the BM25 adapter is `TOTALLY_READY`. To execute the three
> deliverables β€” (Ξ±) native E2E, (Ξ²) controlled-adaptation, (Ξ³) optional
> ablation reference β€” a real dataset is required. This document proposes
> candidates and asks for a decision. Nothing here downloads data.

---

## 1. Constraints

Any chosen dataset must satisfy all of:

| # | Constraint | Why it matters |
|---|---|---|
| 1 | **Pyserini pre-built index available** | Enables byte-for-byte reproduction check |
| 2 | **Published BM25 regression numbers exist** | So we can *prove* our run matches the official pipeline (key for "native") |
| 3 | **Research-friendly license** | No legal blocker for academic publishing |
| 4 | **Domain aligned with SAH-RAG** (long-context / legal, per `docs/report_llm_organizer.md` Β§2) | Comparability with the main architecture |
| 5 | **Size feasible on this hardware** (laptop + GPU server) | Reasonable iteration speed |

---

## 2. Candidates

| Dataset | Docs | Domain | Pyserini index | Published BM25 | License |
|---|---|---|---|---|---|
| **BEIR SciFact** | ~5K | scientific | `beir-v1.0.0-scifact` | nDCG@10 β‰ 0.665 | CC BY-NC 2.0 |
| **BEIR NFCorpus** | ~3.6K | biomedical | `beir-v1.0.0-nfcorpus` | nDCG@10 β‰ 0.325 | CC BY-SA 4.0 |
| **BEIR FiQA** | ~57K | finance | `beir-v1.0.0-fiqa` | nDCG@10 β‰ 0.236 | (mixed, research OK) |
| **MS MARCO passage v1** | 8.8M | web / general | `msmarco-passage` | MRR@10 = 0.187 | research-friendly |
| **MS MARCO passage v2** | 8.8M | web / general | `msmarco-v2-passage` | nDCG@10 β‰ 0.228 (DL19) | research-friendly |
| **Natural Questions (open)** | 2.7M wiki | wikipedia | `wikipedia-dpr` | top-20 β‰ 78% | CC BY-SA |
| **SAH-RAG's chosen dataset** | ? | long-context / legal | TBD | TBD | TBD |

Notes:
- **BEIR subsets** are small (5Kβ€“57K docs) β†’ fast build, useful for
  reproduction check within minutes.
- **MS MARCO passage v1** is the canonical BM25 baseline; every IR paper
  reports on it. Heavy but essential as the reference.
- **Legal / long-context** datasets (e.g., CUAD, ContractNLI, LongBench) do
  **not** have official Pyserini pre-built indexes. They can be used later but
  would not give a reproduction check against published BM25 numbers.

---

## 3. Recommendation β€” three stages

### Stage 1 β€” Reproduction smoke (fast, do first)

**Dataset:** BEIR SciFact (`beir-v1.0.0-scifact`)

**Why:** ~5K docs, index downloads in <1 min, published BM25 nDCG@10 = 0.665
gives a clear pass/fail for "does our adapter reproduce the official pipeline".

**Exit criterion:** our `BM25_NATIVE` run on SciFact matches nDCG@10 = 0.665 Β±
small tolerance (Lucene version drift).

### Stage 2 β€” Native E2E (canonical, real scale)

**Dataset:** MS MARCO passage v1 (`msmarco-passage`)

**Why:** the reference number any IR reviewer will check. Establishes our BM25
baseline as credible for the SAH-RAG paper.

**Exit criterion:** MRR@10 = 0.187 Β± small tolerance on the dev set.

### Stage 3 β€” SAH-RAG-aligned (comparability, deferred)

**Dataset:** whatever the SAH-RAG team picks for their paper.

**Why:** to make BM25 directly comparable to SAH-RAG.

**Status:** deferred until the SAH-RAG team decides. Not my call.

---

## 4. Question for the supervisor

Which path do you want?

- **(a) Start with Stage 1 (SciFact smoke).** Fast validation of the adapter
  against published numbers. Recommended.
- **(b) Skip to Stage 2 (MS MARCO passage v1).** Canonical but heavier download
  (~1-2 GB index, longer build). Skips fast feedback.
- **(c) Wait for Stage 3 (SAH-RAG's dataset).** Blocks all E2E work until the
  SAH-RAG team decides.
- **(d) Different dataset (specify).** e.g., a specific legal dataset if the
  SAH-RAG team already has one in mind.

**My proposal:** (a) first (fast feedback, ~15 min total), then (b) once SciFact
passes, then (c) when SAH-RAG finalizes.

---

## 5. What happens after you decide

The next cell (executed with your answer) will:
1. Download the official Pyserini pre-built index for the chosen dataset.
2. Download the official queries + qrels.
3. Run `BM25_NATIVE` through our adapter.
4. Compute the published metric (nDCG@10 or MRR@10).
5. Compare to the published number and record pass/fail in `reports/`.
6. Produce a `.trec` run file under `baselines/bm25/runs/` for later
   use by stages (Ξ²) and (Ξ³).

No changes to `configs/`, `evaluation/`, `orchestration/`, or SAH-RAG code.
Everything stays inside `baselines/bm25/`.