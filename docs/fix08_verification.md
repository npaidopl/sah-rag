# FIX-08 Verification: LLM Fairness Policy Finalized

**Status:** VERIFIED (2026-11-04)

## Summary

The LLM fairness policy is fully specified in
`docs/llm_fairness_policy.md` + `baseline_audit/llm_fairness_policy.yaml`,
and all required environment variables are documented in `.env.example`
and configured in `.env`.

## Policy state (frozen)

### Controlled adaptation protocol

| Role | Model | Env var |
|---|---|---|
| Indoor LLM | Qwen/Qwen2.5-14B-Instruct | RAG_INDOOR_LLM_ROOT |
| Generation LLM | Qwen/Qwen2.5-32B-Instruct | RAG_GENERATION_LLM_ROOT |
| Generation fallback | Qwen/Qwen2.5-14B-Instruct | RAG_INDOOR_LLM_ROOT |
| Embedder | BAAI/bge-m3 | RAG_EMBEDDER_ROOT (optional) |

### SAH-RAG generation

- LLM: Qwen/Qwen2.5-14B-Instruct
- Mode: temporary (Project 2 will replace)
- Scope: not exercised in IR-only v2

### Fairness enforcement

- Code: `sah_rag.components.llms.QwenLLM`
- Rule: rejects non-14B model_path when `strict_fairness=True`
- Bypass: `strict_fairness=False` (tests/debugging only)

## Baselines coverage (15 total)

### Native E2E ready (5)
- **longrag**: IR-only adapter, embedder=bge-large-en-v1.5 (downloaded)
- **raptor**: Qwen2.5-1.5B local
- **metadata_rag**: IR-only adapter
- **memorag**: Qwen2.5-7B local
- **bm25**: lexical, no model
- **splade**: naver/splade-v3 (deferred download)
- **colbertv2**: colbert-ir/colbertv2.0 (deferred download)

### Native E2E exempt (4, paid API)
- **disretrieval**: architecturally blocked (xlnet backbone missing)
- **lightrag**: GPT-4o-mini required
- **hipporag2**: GPT-4o-mini required
- **multi_meta_rag**: OPENAI_API_KEY required

### Reported only (3, no code)
- **pathrag**: no LICENSE, 0 edges with local LLMs
- **kag**: no public code for KAG-Thinker
- **hikey**: no official code

### Controlled only (1)
- **hikey_controlled**: text-only approximation, NOT HiKEY

## Environment variables

### Required (verified in .env)

- `RAG_CANONICAL_ROOT` → datasets directory
- `RAG_INDOOR_LLM_ROOT` → Qwen 14B (28 GB)
- `RAG_GENERATION_LLM_ROOT` → Qwen 32B (61 GB)

### Optional (documented in .env.example)

- `RAG_EMBEDDER_ROOT` → BGE-M3 (defaults to HF cache)
- `PATH_RAG_REPO` → PathRAG repo (deferred)
- `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `AZURE_OPENAI_*` → native-only

### Verified
RAG_CANONICAL_ROOT: C:\Users\USER1\Documents\SAH-RAG\data\processed\canonical (exists=True)
RAG_INDOOR_LLM_ROOT: D:\rag_work\models\Qwen2.5-14B-Instruct (exists=True)
RAG_GENERATION_LLM_ROOT: D:/rag_work/models/Qwen2.5-32B-Instruct (exists=True)


## Qwen 32B checkpoint

- **Path**: `D:/rag_work/models/Qwen2.5-32B-Instruct`
- **Size**: 61.04 GB (27 files)
- **Shards**: 17/17 (complete)
- **config.json**: present

All shards verified. Model is ready for generation.

## GPU requirements

When GPU arrives, the controlled protocol requires:

| Component | FP16 VRAM | INT8 VRAM | INT4 VRAM |
|---|---|---|---|
| Qwen 14B (indoor) | ~28 GB | ~14 GB | ~7 GB |
| Qwen 32B (generation) | ~64 GB | ~32 GB | ~16 GB |
| BGE-M3 (embedder) | ~4 GB | ~2 GB | ~1 GB |

**If GPU has 40 GB VRAM:**
- Indoor (14B FP16) = 28 GB ✓
- Generation (32B INT8) = 32 GB ✓
- Embedder (BGE-M3 FP16) = 4 GB ✓
- **Total: ~64 GB** — needs sequential loading (unload indoor before generation)

**If GPU has 80 GB VRAM (A100):**
- Indoor (14B FP16) = 28 GB ✓
- Generation (32B FP16) = 64 GB ✓ (but tight)
- Embedder (BGE-M3 FP16) = 4 GB ✓
- **Total: ~96 GB** — needs sequential loading too

**Recommended for run_all.py:**
- Load indoor LLM (14B) → run extraction
- Unload indoor LLM
- Load generation LLM (32B) → run generation
- This is 2× model load time per cell, but no OOM

## Fallback policy

If Qwen 32B cannot be loaded (OOM, missing, etc.):
- Fall back to Qwen 14B (RAG_INDOOR_LLM_ROOT)
- MUST be recorded in `metrics.json` under `generation_model` field
- MUST be flagged in comparison tables
- MUST be disclosed in paper as limitation

## Reference

- Policy: `docs/llm_fairness_policy.md`
- Machine-readable: `baseline_audit/llm_fairness_policy.yaml`
- Env template: `.env.example`

## Change log

- 2026-11-04: added `RAG_GENERATION_LLM_ROOT` to `.env` and `.env.example`
- 2026-11-04: verified env resolution
- 2026-11-04: documented Qwen 32B readiness (61 GB, 17/17 shards)