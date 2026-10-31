"""Native end-to-end harness for LongRAG (Stage B).

This module is NOT part of the SAH-RAG v2 ``BaseBaseline`` contract. It is a
thin orchestration layer that composes:

    LongRAGBaseline  (retrieval, from baselines.longrag_v2.adapter)
  + context builder  (concatenates retrieved long units)
  + a Reader         (GPT-4o / Gemini-1.5-Pro / local sanity)

to produce a native-fidelity end-to-end pipeline matching the LongRAG paper
(arXiv:2406.15319) for the reader stage, which is out of scope for the
IR-only v2 contract.

Fidelity targets:

    - Author-supported native (paper default): GPT4oReader with the official
      prompt template. Requires OPENAI_API_KEY.
    - Author-supported alternative: GeminiReader with the same prompt.
      Requires GOOGLE_API_KEY.
    - Local sanity check ONLY: LocalReader with a small HF model. Never
      makes a native-fidelity claim.

Entry points:

    Python API:
        from baselines.longrag_v2.native_e2e import E2ERunner, load_prompt
        runner = E2ERunner(baseline=..., reader=..., prompt=...)
        results = runner.run_batch(queries)

    CLI:
        python -m baselines.longrag_v2.native_e2e --help
        python -m baselines.longrag_v2.native_e2e.runner --help
"""

from baselines.longrag_v2.native_e2e.cli import (
    build_reader,
    load_baseline_config,
    load_documents_jsonl,
    load_queries_jsonl,
    main,
)
from baselines.longrag_v2.native_e2e.context_builder import build_context
from baselines.longrag_v2.native_e2e.prompt_loader import (
    PromptTemplate,
    load_prompt,
    render,
)
from baselines.longrag_v2.native_e2e.reader import (
    GPT4oReader,
    GeminiReader,
    LocalReader,
    Reader,
)
from baselines.longrag_v2.native_e2e.runner import E2EResult, E2ERunner

__all__ = [
    # Readers
    "Reader",
    "GPT4oReader",
    "GeminiReader",
    "LocalReader",
    # Runner
    "E2ERunner",
    "E2EResult",
    # Building blocks
    "build_context",
    "load_prompt",
    "render",
    "PromptTemplate",
    # CLI + config loading
    "main",
    "build_reader",
    "load_baseline_config",
    "load_documents_jsonl",
    "load_queries_jsonl",
]