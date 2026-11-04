"""End-to-end SAH-RAG demo (offline by default, no Qwen weights needed).

Usage:
    # Offline (dry-run adapter):
    python examples/sah_rag_e2e_demo.py

    # With real Qwen (requires weights + env var):
    $env:RAG_INDOOR_LLM_ROOT = "D:/rag_work/models/Qwen2.5-14B-Instruct"
    python examples/sah_rag_e2e_demo.py --use-qwen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from sah_rag.generation import (  # noqa: E402
    DeterministicDryRunAdapter,
    PipelineConfig,
    QwenLLMAdapter,
    answer,
)
from sah_rag.retrieval import RetrievedChunk  # noqa: E402


def _make_demo_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            parent_id="parent-1",
            doc_id="doc-1",
            dataset="clerk",
            rank=1,
            score=0.95,
            text_normalized=(
                "Under section 12 of the statute, the maximum penalty "
                "for the offense is ten years imprisonment and a fine "
                "of up to one hundred thousand dollars."
            ),
            retrieval_method="hybrid_rrf",
            payload={"source": "legal_statute"},
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            parent_id="parent-2",
            doc_id="doc-2",
            dataset="clerk",
            rank=2,
            score=0.87,
            text_normalized=(
                "Section 15 clarifies that the penalty applies to "
                "individuals as well as legal entities."
            ),
            retrieval_method="hybrid_rrf",
            payload={"source": "legal_statute"},
        ),
        RetrievedChunk(
            chunk_id="chunk-3",
            parent_id="parent-3",
            doc_id="doc-3",
            dataset="clerk",
            rank=3,
            score=0.72,
            text_normalized=(
                "The statute defines 'offense' in section 3, "
                "referring to violations of sections 5 through 11."
            ),
            retrieval_method="hybrid_rrf",
            payload={"source": "legal_statute"},
        ),
    ]


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_result(result) -> None:
    _print_header("Generated Answer")
    print(f"answer_id:  {result.answer_id}")
    print(f"prompt_id:  {result.prompt_id}")
    print(f"dataset:    {result.dataset_id}")
    print(f"query_id:   {result.query_id}")
    print(f"model:      {result.generation_model_id}")
    print(f"runtime:    {result.generation_runtime}")
    print(f"tokens in:  {result.input_token_count}")
    print(f"tokens out: {result.output_token_count}")
    print()
    print("Answer text:")
    print("-" * 70)
    print(result.answer_text)
    print("-" * 70)
    print()
    print("Citation validation:")
    cv = result.citation_validation
    print(f"  valid:               {cv.valid}")
    print(f"  has_citations:       {cv.has_citations}")
    print(f"  all_citations_known: {cv.all_citations_known}")
    print(f"  available:           {cv.available_citation_ids}")
    print(f"  used:                {cv.used_citation_ids}")
    print(f"  unknown:             {cv.unknown_citation_ids}")
    print(f"  unused:              {cv.unused_citation_ids}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SAH-RAG end-to-end demo")
    parser.add_argument(
        "--use-qwen",
        action="store_true",
        help="Use Qwen2.5-14B-Instruct instead of dry-run adapter",
    )
    args = parser.parse_args()

    _print_header("SAH-RAG End-to-End Demo")
    print("Mode:", "Qwen2.5-14B-Instruct" if args.use_qwen else "DeterministicDryRunAdapter")

    chunks = _make_demo_chunks()
    print(f"\n[1/3] Retrieved {len(chunks)} chunks:")
    for c in chunks:
        print(f"  rank={c.rank} score={c.score:.2f} chunk_id={c.chunk_id}")

    if args.use_qwen:
        runtime = QwenLLMAdapter()
    else:
        dry_answer = (
            "Based on section 12 [C1], the maximum penalty is ten years "
            "imprisonment and a fine up to one hundred thousand dollars. "
            "Section 15 [C2] confirms the penalty applies to both "
            "individuals and legal entities."
        )
        runtime = DeterministicDryRunAdapter(answer_text=dry_answer)

    config = PipelineConfig()
    print(f"\n[2/3] Pipeline config:")
    print(f"  maximum_context_tokens: {config.maximum_context_tokens}")
    print(f"  maximum_prompt_tokens:  {config.maximum_prompt_tokens}")
    print(f"  reserved_answer_tokens: {config.reserved_answer_tokens}")

    print(f"\n[3/3] Running pipeline...")
    try:
        result = answer(
            query_id="demo-q-001",
            query_text="What is the maximum penalty for the offense?",
            dataset_id="clerk",
            retrieved_chunks=chunks,
            runtime=runtime,
            config=config,
        )
    except Exception as exc:
        print(f"\n[FAILED] {type(exc).__name__}: {exc}")
        return 1

    _print_result(result)

    if result.citation_validation.valid:
        print("\n[SUCCESS] End-to-end RAG pipeline works.")
        return 0
    else:
        print("\n[WARNING] Answer generated but citations are invalid.")
        return 2


if __name__ == "__main__":
    sys.exit(main())