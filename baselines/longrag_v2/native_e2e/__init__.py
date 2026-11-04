"""Native end-to-end harness for LongRAG (Stage B).

Composes:
    LongRAGBaseline + two-step reader (long answer -> short answer)

Native fidelity targets supported (per baseline_audit/longrag_native_reader_config.yaml):

    - NQ (paper default reader: GPT-4o)
    - HotpotQA (same reader)
    - Qasper / MultiFieldQA-en are NOT supported as native because the
      official repository does not implement them. Any attempt to run
      them via this module would be a controlled adaptation, not native.

Entry points:

    Python API:
        from baselines.longrag_v2.native_e2e import E2ERunner, load_prompt
        runner = E2ERunner(
            baseline=..., reader=...,
            main_prompt=load_prompt("nq"),
            extraction_prompt=load_prompt("answer_extraction"),
            dataset_name="nq",
        )

    CLI:
        python -m baselines.longrag_v2.native_e2e --help
"""

from baselines.longrag_v2.native_e2e.cli import (
    build_reader,
    load_baseline_config,
    load_documents_jsonl,
    load_queries_jsonl,
    main,
)
from baselines.longrag_v2.native_e2e.context_builder import (
    BuiltContext,
    build_context,
)
from baselines.longrag_v2.native_e2e.eval_util import (
    compute_exact,
    has_correct_answer,
    normalize_answer,
    single_ans_em,
)
from baselines.longrag_v2.native_e2e.prompt_loader import (
    PromptTemplate,
    load_demo_examples,
    load_prompt,
    regenerate_demos,
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
    "BuiltContext",
    "build_context",
    "load_prompt",
    "render",
    "PromptTemplate",
    "load_demo_examples",
    "regenerate_demos",
    # Metrics
    "single_ans_em",
    "compute_exact",
    "normalize_answer",
    "has_correct_answer",
    # CLI + config loading
    "main",
    "build_reader",
    "load_baseline_config",
    "load_documents_jsonl",
    "load_queries_jsonl",
]