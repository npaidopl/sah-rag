# BM25 → SAH-RAG Ablation: Idea Proposal

**Status:** Idea only. For the SAH-RAG team to evaluate with their main LLM.
**Date:** 2026-10-23

This document proposes **how BM25 could participate in the SAH-RAG ablation
study**, given that SAH-RAG is a structure-aware hierarchical RAG. The BM25
baseline itself is complete; these are suggestions for how the SAH-RAG team
might use it as an ablation component.

---

## Constraint

The BM25 baseline is a **lexical retriever**. It does not generate answers,
does not embed densely, and has no hierarchical structure awareness. Therefore
it can only participate in an ablation as a **retrieval component**, not as a
generation component.

---

## Idea A — BM25 as first-stage candidate retriever for SAH-RAG

**What:** Use BM25 as a cheap lexical retriever to fetch top-N candidate
documents/chunks, then let SAH-RAG's hierarchical traversal handle the rest.

**Ablation arm:** `SAH-RAG − BM25_first_stage` (replace BM25 with random
sampling or dense-only candidate retrieval)

**Pros:**
- BM25 runs at 300+ q/s steady state; adds negligible latency
- If SAH-RAG is dense-only first stage, BM25 adds lexical recall (complementary)
- Well-established baseline in IR; easy to interpret

**Cons:**
- Only useful if SAH-RAG has a "candidate retrieval" phase distinct from
  hierarchical ranking. If SAH-RAG does end-to-end hierarchical traversal
  without explicit first-stage retrieval, this does not apply.

**Data needed:** ablation table row + comparison of SAH-RAG with/without
BM25 first stage.

**Effort for SAH-RAG team:** medium — requires architecture change to insert
BM25 as first-stage component.

---

## Idea B — BM25 score as a feature in SAH-RAG re-ranking

**What:** BM25 score (or top-k BM25 rank) fused into SAH-RAG's re-ranking
score as an additional feature.

**Ablation arm:** `SAH-RAG − BM25_feature` (drop the BM25 feature from the
re-ranking formula)

**Pros:**
- Minimal architecture change: only the reranker scoring function changes
- Quantifiable contribution: A/B without changing pipeline structure
- Standard IR technique (score fusion)

**Cons:**
- Requires SAH-RAG to expose a reranker with feature slots
- Fusion weights need tuning (could confound ablation)

**Data needed:** ablation table row showing SAH-RAG with/without BM25 score
feature in reranker.

**Effort for SAH-RAG team:** low-medium — depends on reranker extensibility.

---

## Idea C — BM25+RM3 query expansion as pre-processing step

**What:** Use BM25+RM3 (pseudo-relevance feedback) to rewrite the query
before SAH-RAG's hierarchical traversal. RM3 expands the query with terms
from top BM25 results.

**Ablation arm:** `SAH-RAG − RM3_expansion`

**Pros:**
- Purely at input layer — SAH-RAG architecture unchanged
- Well-documented Pyserini feature (`--bm25 --rm3`)
- Complements any downstream retrieval

**Cons:**
- Adds latency (~2x retrieval cost)
- Effect depends on whether SAH-RAG already does query rewriting

**Data needed:** ablation table row showing SAH-RAG with/without RM3-
expanded queries.

**Effort for SAH-RAG team:** low — just change input query, run BM25+RM3,
pass expanded query to SAH-RAG.

---

## Idea D — BM25 score as routing signal

**What:** For SAH-RAG's hierarchical traversal, use BM25 score variance
as a signal to decide when to descend into a subtree vs. traverse sideways.

**Ablation arm:** `SAH-RAG − BM25_routing`

**Pros:**
- Uses BM25 as a cheap discriminator
- Potential for meaningful gains if SAH-RAG's routing is currently heuristic

**Cons:**
- Requires SAH-RAG to have a routing layer that accepts external signals
- Speculative — depends on SAH-RAG's design

**Data needed:** ablation table row showing SAH-RAG with/without BM25
routing signal.

**Effort for SAH-RAG team:** medium-high — requires routing integration.

---

## Recommendation

**Idea B (BM25 as re-ranking feature) is the most tractable** because:
- It requires the smallest architecture change
- It produces a clean ablation (feature on/off)
- It does not require BM25 to be "integrated" as a submodule

**Idea C (RM3 query expansion) is the lowest-effort** but likely offers
smaller gains.

**Idea A (first-stage candidate retriever)** is architecturally natural if
SAH-RAG already has a candidate retrieval phase; if not, skip.

**Idea D (routing signal)** is speculative and depends on SAH-RAG's
internal design.

---

## What this document is not

- Not a commitment: no code will be written by the BM25 baseline for these
  ideas.
- Not a plan: the SAH-RAG team decides whether to pursue any of them.
- Not a test: the SAH-RAG team's main LLM should evaluate these ideas
  alongside other ablation proposals.

## If SAH-RAG team selects an idea

Once selected, the BM25 baseline can provide:
- The `BM25_REFERENCE` row (already available: baseline retrieval metrics)
- Support for `BM25+RM3` configuration if Idea C is chosen (requires
  adding an RM3 config — small change, ~1 hour)
- No further adapter work otherwise.

---

## Provenance

- BM25 baseline native-fidelity: **PAPER_EXACT_NATIVE** (SciFact)
- BM25 baseline controlled-adaptation: **verified with real BGE-M3**
- Full evidence: `baselines/bm25/FINAL_REPORT.md`