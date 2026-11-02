# LLM Fairness Policy - v2

**Status:** FROZEN (v1.0, 2026-11-02)
**Scope:** IR-only (v2). E2E boundary in native mode = retrieved ranked list, not generated answer.

---

## Purpose

Define which LLM is used by each baseline in each evaluation mode:

- **native E2E**: baseline uses its own native components (embedder, LLM for entity extraction/summarization, index, retriever)
- **controlled**: shared embedder + shared LLM injected by the orchestrator
- **ablation**: variant runs of the SAH-RAG architecture (covered in ablation docs, not here)

This policy ensures:

1. **Fairness in controlled mode**: all baselines face identical component constraints
2. **Fidelity in native mode**: each baseline reproduces as closely as possible to paper/author-supported configuration
3. **Transparency**: documented exemptions when native fidelity is impossible (paid API, license, missing code)

---

## Section 1: Native E2E Rules

**Rule 1.1:** Each baseline runs with its OWN native components. No injection.

**Rule 1.2:** "Native LLM" = the LLM used by the baseline in its RETRIEVAL pipeline (not answer generation). Examples: entity extraction (HippoRAG2), cluster summarization (RAPTOR), OpenIE (LightRAG).

**Rule 1.3:** If the paper uses a paid LLM API (GPT-4o, GPT-3.5-turbo, Gemini, etc.) and no free/local author-supported alternative exists -> exempt from native E2E. Baseline runs only in controlled mode.

**Rule 1.4:** If the baseline has no official code, no public repo, or is architecturally blocked (missing backbone, unlicensed weights) -> reported only (paper numbers, no execution).

**Rule 1.5:** If v2 is IR-only and the baseline is not LLM-based (e.g. BM25, SPLADE, ColBERTv2, Metadata-RAG) -> native E2E means IR-only E2E. No LLM required.

### Native E2E verdict table

| Baseline | Status | Modes | LLM-based | Native LLM | Native E2E |
|---|---|---|---|---|---|
| disretrieval | executable | controlled | yes | (BLOCKED) | EXEMPT: missing xlnet backbone |
| pathrag | reported | - | yes | GPT-4o-mini | REPORTED |
| longrag | executable | native, controlled | no | bge-large-en-v1.5 | YES |
| lightrag | executable | native, controlled | yes | GPT-4o-mini | EXEMPT: paid API |
| hipporag2 | executable | native, controlled | yes | GPT-4o-mini | EXEMPT: paid API |
| raptor | executable | native, controlled | yes | Qwen2.5-1.5B (local) | YES (author-supported) |
| metadata_rag | executable | native, controlled | no | - | YES |
| memorag | executable | native, controlled | yes | Qwen2.5-7B (local) | YES (author-supported) |
| kag | reported | - | yes | KAG-Thinker-7B | REPORTED |
| bm25 | executable | native | no | - | YES |
| splade | executable | native, controlled | no | naver/splade-v3 | YES (deferred download) |
| colbertv2 | executable | native | no | colbert-ir/colbertv2.0 | YES (deferred download) |
| multi_meta_rag | executable | native, controlled | no | GPT-3.5/GPT-4 | EXEMPT: paid API |
| hikey | reported | - | no | (no code) | REPORTED |
| hikey_controlled | executable | controlled | no | - | CONTROLLED-ONLY (approximation) |

Legend:

- **YES**: native E2E runs locally, no external dependency
- **EXEMPT**: cannot run natively (see Section 4)
- **REPORTED**: paper-reported metrics only; no execution
- **CONTROLLED-ONLY**: adapter supports only controlled mode by design

---

## Section 2: Controlled Protocol

**Applies to:** all baselines with `is_llm_based: true` (LLM injected) and any baseline in `controlled` mode (embedder injected).

**Shared components:**

| Component | Model | Purpose |
|---|---|---|
| Embedder | `BAAI/bge-m3` (dense + sparse) | injected into every controlled-mode run |
| Indoor LLM | `Qwen/Qwen2.5-14B-Instruct` | used by LLM-based baselines for entity extraction / cluster summarization / OpenIE inside their retrieval pipeline |
| Generation LLM (SAH-RAG only) | `Qwen/Qwen2.5-14B-Instruct` | see Section 3 |

**Rule 2.1:** All controlled-mode runs use the SAME embedder (BGE-M3). No per-baseline embedder override.

**Rule 2.2:** LLM-based baselines must call `self.llm.generate()` when they need LLM inference in their pipeline. No external API calls.

**Rule 2.3 (fallback, VERBATIM):**

> If `Qwen2.5-32B-Instruct` is unavailable at evaluation time, generation falls back to `Qwen2.5-14B-Instruct`. This MUST be (a) recorded in each run's `metrics.json` under `generation_model` field, (b) flagged in any comparison table, and (c) disclosed in the paper as a limitation.

**Note:** The 32B generation LLM is downloaded (`D:\rag_work\models\Qwen2.5-32B-Instruct`, 61.04 GB, verified 2026-11-02). Fallback only applies if the 32B is unavailable at runtime.

**Rule 2.4:** Fairness protocol enforcement. `sah_rag.components.llms.QwenLLM` refuses to initialize with a non-14B model unless `strict_fairness=False` is explicit (tests/debugging only). Enforced at code level.

---

## Section 3: SAH-RAG Generation

The SAH-RAG architecture itself (not a baseline) uses:

- **Generation LLM:** `Qwen2.5-14B-Instruct` (temporary selection; Project 2 will replace)
- **Prompt:** standard RAG prompt (defined in `sah_rag/generation/prompts.py`)

This is IR-only v2 scope, so generation is not exercised in current evaluation. Generation will be activated in Project 2.

---

## Section 4: Exemptions & Rationale

Baselines listed below are exempt from native E2E mode. Each has a specific reason.

### 4.1 Paid-API block (paper uses paid LLM)

| Baseline | Paper LLM | Alternative | Status |
|---|---|---|---|
| lightrag | GPT-4o-mini | none documented | EXEMPT |
| hipporag2 | GPT-4o-mini | none documented | EXEMPT |
| multi_meta_rag | GPT-3.5/GPT-4 | none documented | EXEMPT |

These baselines require an external paid API for the native pipeline. They run only in controlled mode where the shared `Qwen2.5-14B-Instruct` is injected.

### 4.2 Architecturally blocked

| Baseline | Reason |
|---|---|
| disretrieval | Official repo ships parser head weights but omits the fine-tuned XLNet backbone (`xlm_rst_tuned/`). Native mode cannot load. Controlled mode uses upstream E-NNRSTParser (Apache-2.0) + DISRetrieval architecture logic. |

### 4.3 Reported-only (no public code)

| Baseline | Disposition |
|---|---|
| pathrag | Wrapper blocked: no LICENSE in official repo, 0 edges with available local LLMs, mock LLM violated contract. Metrics from paper only. |
| kag | No public code for the KAG-Thinker system. Metrics from paper only. |
| hikey | No official code exists. Metrics from paper only (M3DocVQA, FRAMES-PDF). |

### 4.4 Controlled-only (by design)

| Baseline | Reason |
|---|---|
| hikey_controlled | Text-only approximation. NOT HiKEY. Paper positioning: footnote, not separate row. Runs in controlled mode only (injected BGE-M3). Approximation marker in `RetrievedChunk.metadata`. |

### 4.5 Deferred native components (downloadable)

| Baseline | Component | Status |
|---|---|---|
| splade | `naver/splade-v3` (CC BY-NC-SA 4.0) | academic use only; download when native E2E runs |
| colbertv2 | `colbert-ir/colbertv2.0` (MIT) | download when native E2E runs |
| longrag | `bge-large-en-v1.5` | downloaded 2026-11-02 (2.56 GB) |

### 4.6 Note on multi_meta_rag

Multi_Meta_RAG native requires `OPENAI_API_KEY` (paid) and `bge-large-en-v1.5`. The embedder is now downloaded, but the paid API requirement means native E2E remains EXEMPT. Controlled mode is fully ready with injected BGE-M3.

---

## Section 5: Machine-readable companion

`baseline_audit/llm_fairness_policy.yaml` - machine-readable version of this policy. Any consumer (orchestrator, reporter, audit scripts) should read the YAML, not parse this document.

---

## Changelog

| Date | Version | Change |
|---|---|---|
| 2026-11-02 | 1.0 | Initial policy. Covers all 15 registered baselines. |

