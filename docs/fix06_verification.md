# FIX-06 Verification: LightRAG Controlled Mode

**Status:** Infrastructure VERIFIED / Execution GPU-DEFERRED (2026-11-04)

## Summary

LightRAG controlled mode is fully wired at the code level but requires
the Qwen2.5-14B-Instruct model (~28 GB) for real inference. The current
development laptop has only 15.4 GB RAM, so execution is deferred to
the university GPU server.

## Test results (15/15 passed in 1.25s)

### Core adapter behavior
- test_build_and_retrieve_native
- test_empty_corpus
- test_top_k_respected
- test_retrieve_before_build

### Controlled-mode contracts
- test_controlled_with_mocks (uses mock LLM + mock embedder)
- test_controlled_requires_components (fails fast if no injection)
- test_full_pipeline_raises_without_components

### Full pipeline state
- test_retrieve_full_raises_without_build
- test_strict_full_path_raises_at_build (fail-fast works)
- test_retrieve_uses_full_when_full_rag_set

### Parser strategies (4 strategies for robustness)
- test_parse_context_substring_match
- test_parse_context_reference_id
- test_parse_context_partial_text

### Event loop management
- test_persistent_event_loop_reused
- test_cleanup_stops_loop

## Controlled mode requirements

Looking at `baselines/lightrag/adapter.py`:

| Component | Required? | Injected by runner? |
|---|---|---|
| `self.embedder` (BGE-M3) | YES | YES (`_build_controlled_components`) |
| `self.llm` (Qwen 14B) | YES (is_llm_based=True) | YES |

Controlled mode logic at L281, L358, L377, L476 checks both are injected
and uses `self.llm.generate()` for LLM-based extraction inside the pipeline.

## Why GPU is required

LightRAG's controlled pipeline uses the injected LLM for:
1. **Entity extraction** (per chunk)
2. **Relationship extraction** (per chunk)
3. **Keyword extraction** (per query)

Each requires a full forward pass through Qwen 14B. The model needs
~28 GB in FP16. Local hardware (15.4 GB RAM) cannot load it.

**Mitigation options considered:**
- Mock LLM for integration testing → **already covered by test_controlled_with_mocks**
- Smaller Qwen (1.5B) with `strict_fairness=False` → **not acceptable for research**, only for plumbing tests
- CPU inference with disk paging → **estimated 2-5 hours per query**, not viable

## What is proven (offline)

1. **Adapter structure:** The 15 tests confirm the adapter's contract compliance.
2. **Runner dispatch:** `_build_controlled_components` correctly injects BGE-M3 + QwenLLM for LLM-based baselines.
3. **Fail-fast behavior:** `strict_full_path=true` in production config raises immediately if Qwen is unavailable — no silent fallback that would misrepresent native-ness.
4. **Event loop safety:** The persistent event loop from the native acceptance test (commit b7b7593) still works in controlled mode.
5. **Registry integration:** The `baselines_registry.yaml` entry has `status: executable`, `modes: [native, controlled]`, and both config files.

## What is deferred (GPU-dependent)

1. **Real end-to-end controlled run** on `infrastructure-pilot-synthetic-v1`.
2. **Comparison** with native mode (native uses Qwen 1.5B, controlled uses Qwen 14B — different models but same architecture).
3. **Metrics collection** — until GPU access is available.

## How to run when GPU arrives

    # 1. Verify Qwen 14B is on disk
    Test-Path "D:\rag_work\models\Qwen2.5-14B-Instruct"

    # 2. Set env var (from .env, already configured)
    $env:RAG_INDOOR_LLM_ROOT = "D:\rag_work\models\Qwen2.5-14B-Instruct"

    # 3. Run controlled mode
    python -m orchestration.scheduler configs/pilot_lightrag_controlled.yaml

    # 4. Expected output:
    #    - outputs/<run_id>/metrics.json
    #    - outputs/<run_id>/predictions.jsonl

No code changes needed. The manifest is already provisioned.

## Files

- Adapter: `baselines/lightrag/adapter.py` (34 KB)
- Config: `baselines/lightrag/config_controlled.yaml`
- Tests: `baselines/lightrag/tests/test_adapter.py` (15 tests)
- Registry: `configs/baselines_registry.yaml` (entry `lightrag`)
- Runner dispatch: `evaluation/runner.py::_build_controlled_components`

## Impact on run_all.py

The orchestrator will treat LightRAG controlled cells as:
- **enabled: true** (infrastructure is ready)
- **requires_gpu: true** (Qwen 14B mandatory)
- **requires_env: RAG_INDOOR_LLM_ROOT**

When the orchestrator runs on GPU, these cells execute normally.
When run on local CPU, the orchestrator should skip them with a clear
message (via `run_all.py` preflight check on VRAM + model availability).

## Change log

- 2026-11-04: verified infrastructure (FIX-06). Execution deferred to GPU.