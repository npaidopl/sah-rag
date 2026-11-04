# SAH-RAG Generation Pipeline (temporary, v1)

## Purpose

Turns SAH-RAG from pure IR into end-to-end RAG. **NOT the final
generation architecture** -- Project 2 will replace it with
LLM+agent+scratchpad for long-context, multi-hop, dynamic retrieval.

## Pipeline

    RetrievedChunk -> build_context_pack -> ContextPack
                  -> build_generation_prompt -> GenerationPrompt
                  -> QwenLLMAdapter.generate() -> GenerationRuntimeResponse
                  -> build_generated_answer -> GeneratedAnswer

## Public API

    from sah_rag.generation import (
        answer,              # end-to-end
        PipelineConfig,      # token budgets
        QwenLLMAdapter,      # production runtime
        DeterministicDryRunAdapter,  # offline test
        GeneratedAnswer,
    )

### Minimal usage

    from sah_rag.generation import answer

    result = answer(
        query_id="clerk-q-001",
        query_text="What is the maximum penalty?",
        dataset_id="clerk",
        retrieved_chunks=chunks,
    )
    print(result.answer_text)
    print(result.citation_validation.valid)

### Offline testing

    from sah_rag.generation import DeterministicDryRunAdapter

    result = answer(
        query_id="q-1",
        query_text="?",
        dataset_id="clerk",
        retrieved_chunks=chunks,
        runtime=DeterministicDryRunAdapter(answer_text="X [C1]."),
    )

## Token Budgets (Qwen 2048-token ceiling)

| Param | Default | Rationale |
|---|---|---|
| maximum_context_tokens | 1400 | Sources fit under 2048 |
| maximum_prompt_tokens | 2048 | Qwen max_length hardcoded |
| reserved_answer_tokens | 400 | Answer headroom |
| tokenizer_id | word-counter-v1 | Approximation |

## Limitations

1. **Single-step.** No multi-hop, no query rewriting.
2. **No agent / scratchpad.**
3. **Approximate token counter.** word-counter-v1 under-estimates Qwen by ~20%; pipeline leaves 400-token headroom.
4. **No streaming.** Synchronous generate().
5. **Mandatory Qwen 14B.** QwenLLM(strict_fairness=True) enforces.
6. **Not used by IR-only baselines.** Optimal Operation, Ablation, Controlled Adaptation are retrieval-only. Only Native E2E baselines and paper demonstration use this.

## vs Project 2

| Feature | v1 (this) | v2 (Project 2) |
|---|---|---|
| Context assembly | YES | YES |
| Citation validation | YES | YES |
| Multi-hop | NO | YES |
| Query rewriting | NO | YES |
| Scratchpad | NO | YES |
| Agent orchestration | NO | YES |
| Streaming | NO | YES |

## Design principles (preserved from prior session)

1. Deterministic artifacts (content-addressable IDs)
2. Typed Pydantic models everywhere
3. Citation integrity enforced (answers must cite sources)
4. No silent truncation (budgets enforced at build time)
5. Provider-neutral runtime (GenerationRuntimeAdapter protocol)

## Files

- sah_rag/generation/models.py -- ContextItem, ContextPack
- sah_rag/generation/context.py -- build_context_pack
- sah_rag/generation/prompts.py -- build_generation_prompt
- sah_rag/generation/answers.py -- GeneratedAnswer + validation
- sah_rag/generation/runtime.py -- GenerationRuntimeAdapter protocol
- sah_rag/generation/qwen_adapter.py -- QwenLLMAdapter (this session)
- sah_rag/generation/pipeline.py -- answer() (this session)
- tests/unit/test_qwen_adapter.py -- 6 tests
- tests/unit/test_generation_pipeline.py -- 9 tests

## Change log

- 2026-11-04: added QwenLLMAdapter + pipeline.answer() (this session)