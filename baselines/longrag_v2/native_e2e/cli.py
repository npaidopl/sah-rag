"""Command-line entry point for the LongRAG native end-to-end runner.

Usage
-----
::

    python -m baselines.longrag_v2.native_e2e \
        --docs      path/to/documents.jsonl \
        --queries   path/to/queries.jsonl \
        --baseline-config baselines/longrag_v2/config_native.yaml \
        --reader-config   baselines/longrag_v2/native_e2e/configs/reader_gpt4o.yaml \
        --prompt    nq \
        --out       outputs/longrag_native/nq \
        [--top-k 10] [--max-context-chars 120000] \
        [--limit 5] [--seed 42] [--dry-run]

The same entry point works via ``python -m baselines.longrag_v2.native_e2e.runner``.

Outputs
-------
``<out>/index/``        adapter index artifacts (vectors, records, meta)
``<out>/manifest.json`` full provenance + per-query results
``<out>/answers.jsonl`` one {query_id, answer, duration_ms} per line

Notes
-----
* Native mode is the default. Controlled mode requires an injected embedder
  and is not exposed through this CLI; use the Python API or the
  orchestration framework for controlled comparisons.
* The runner loads the entire corpus into memory (the adapter does not
  stream). For the SAH-RAG canonical suite this is fine; for NQ-full it is
  not (22M documents).
* API keys are read from the environment by the reader implementations
  (``OPENAI_API_KEY`` / ``GOOGLE_API_KEY``). This CLI does not accept them
  as flags, to avoid leaking secrets into shell history.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from baselines.base import BaselineConfig, RunMode
from baselines.longrag_v2.adapter import LongRAGBaseline
from baselines.longrag_v2.native_e2e.eval_util import (
    has_correct_answer,
    single_ans_em,
)
from baselines.longrag_v2.native_e2e.prompt_loader import (
    load_demo_examples,
    load_prompt,
)
from baselines.longrag_v2.native_e2e.reader import (
    GPT4oReader,
    GeminiReader,
    LocalReader,
    Reader,
)
from baselines.longrag_v2.native_e2e.runner import E2ERunner
from sah_rag.canonical.models import CanonicalDocument, CanonicalQuery


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_documents_jsonl(path: str | Path) -> list[CanonicalDocument]:
    """Load one CanonicalDocument per JSONL line."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Documents file not found: {p}")
    docs: list[CanonicalDocument] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                docs.append(CanonicalDocument(**data))
            except Exception as exc:
                raise ValueError(
                    f"Invalid document at {p}:{i}: {exc}"
                ) from exc
    if not docs:
        raise ValueError(f"No documents loaded from {p}")
    return docs


def load_queries_jsonl(path: str | Path) -> list[CanonicalQuery]:
    """Load one CanonicalQuery per JSONL line."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Queries file not found: {p}")
    queries: list[CanonicalQuery] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                queries.append(CanonicalQuery(**data))
            except Exception as exc:
                raise ValueError(
                    f"Invalid query at {p}:{i}: {exc}"
                ) from exc
    if not queries:
        raise ValueError(f"No queries loaded from {p}")
    return queries


def load_baseline_config(
    path: str | Path,
    *,
    top_k_override: int | None = None,
) -> BaselineConfig:
    """Load BaselineConfig from a YAML file matching the frozen v2 schema."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Baseline config must be a mapping: {p}")
    top_k = (
        top_k_override
        if top_k_override is not None
        else int(data.get("top_k", 10))
    )
    return BaselineConfig(
        name=str(data.get("name", "longrag")),
        mode=RunMode(str(data.get("mode", "native"))),
        top_k=top_k,
        extra=dict(data.get("extra") or {}),
    )


def build_reader(path: str | Path) -> Reader:
    """Instantiate a Reader from a YAML config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reader config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Reader config must be a mapping: {p}")
    cfg = data.get("reader", data)
    rtype = str(cfg.get("type", "gpt4o")).lower()

    if rtype == "gpt4o":
        return GPT4oReader(
            model=str(cfg.get("model", "gpt-4o")),
            temperature=float(cfg.get("temperature", 0.0)),
            seed=int(cfg.get("seed", 12345)),
            max_tokens=int(cfg.get("max_tokens", 1000)),
            retry=int(cfg.get("retry", 3)),
            delay_seconds=float(cfg.get("delay_seconds", 5.0)),
            azure_endpoint=cfg.get("azure_endpoint"),
            azure_api_version=cfg.get("azure_api_version"),
        )
    if rtype == "gemini":
        return GeminiReader(
            model=str(cfg.get("model", "gemini-1.5-pro")),
            temperature=float(cfg.get("temperature", 0.0)),
            max_output_tokens=int(cfg.get("max_output_tokens", 1000)),
        )
    if rtype == "local":
        return LocalReader(
            model_name=str(cfg.get("model", "Qwen/Qwen2.5-0.5B-Instruct")),
            max_new_tokens=int(cfg.get("max_new_tokens", 256)),
            device=str(cfg.get("device", "auto")),
        )
    raise ValueError(
        f"Unknown reader type {rtype!r}. Expected one of: gpt4o, gemini, local."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m baselines.longrag_v2.native_e2e",
        description="LongRAG native end-to-end runner (Stage B).",
    )
    p.add_argument("--docs", required=False, help="Path to documents.jsonl")
    p.add_argument("--queries", required=False, help="Path to queries.jsonl")
    p.add_argument(
        "--baseline-config",
        dest="baseline_config",
        required=False,
        help="Path to config_native.yaml",
    )
    p.add_argument(
        "--reader-config",
        dest="reader_config",
        required=False,
        help="Path to reader_*.yaml",
    )
    p.add_argument(
        "--prompt",
        required=False,
        help="Main prompt template name (nq or hotpotqa) or explicit .txt path. "
        "Qasper and MultiFieldQA-en are NOT supported as native; they have no "
        "official prompts in the LongRAG repository.",
    )
    p.add_argument(
        "--extraction-prompt",
        dest="extraction_prompt",
        default="answer_extraction",
        help="Extraction prompt template name (default: answer_extraction).",
    )
    p.add_argument(
        "--dataset-name",
        dest="dataset_name",
        default=None,
        help="Dataset identifier for the manifest (e.g. 'nq', 'hotpotqa'). "
        "Defaults to the value of --prompt.",
    )
    p.add_argument(
        "--demos",
        default=None,
        help="Path to a demo examples file (default: use HF cache at "
        "native_e2e/prompts/answer_extraction_demos.txt).",
    )
    p.add_argument(
        "--out",
        required=False,
        help="Output directory for index, manifest, answers",
    )
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-context-chars", type=int, default=120_000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate all inputs, print a summary, do not run.",
    )
    return p


def _require(args: argparse.Namespace, names: list[str]) -> list[str]:
    missing = [n for n in names if getattr(args, n) is None]
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    missing = _require(
        args,
        ["docs", "queries", "baseline_config", "reader_config", "prompt", "out"],
    )
    if missing:
        print(
            f"error: missing required arguments: {', '.join(missing)}",
            file=sys.stderr,
        )
        parser.print_usage(sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_dir = out_dir / "index"

    try:
        docs = load_documents_jsonl(args.docs)
        queries = load_queries_jsonl(args.queries)
        if args.limit is not None and args.limit > 0:
            queries = queries[: args.limit]
        baseline_cfg = load_baseline_config(
            args.baseline_config, top_k_override=args.top_k
        )
        reader = build_reader(args.reader_config)
        main_prompt = load_prompt(args.prompt)
        extraction_prompt = load_prompt(args.extraction_prompt)
        if args.demos:
            demos = Path(args.demos).read_text(encoding="utf-8")
        else:
            demos = load_demo_examples()
        dataset_name = args.dataset_name or main_prompt.name
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if baseline_cfg.mode != RunMode.NATIVE:
        print(
            "error: this CLI only runs native mode. "
            f"baseline config has mode={baseline_cfg.mode.value!r}.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        summary = {
            "status": "dry_run",
            "num_documents": len(docs),
            "num_queries": len(queries),
            "dataset_name": dataset_name,
            "baseline": {
                "name": baseline_cfg.name,
                "mode": baseline_cfg.mode.value,
                "top_k": baseline_cfg.top_k,
                "grouping_policy": baseline_cfg.extra.get("grouping_policy"),
            },
            "reader": {"name": reader.name},
            "main_prompt": {
                "name": main_prompt.name,
                "sha256": main_prompt.sha256,
            },
            "extraction_prompt": {
                "name": extraction_prompt.name,
                "sha256": extraction_prompt.sha256,
            },
            "demos_length_chars": len(demos),
            "out_dir": str(out_dir),
        }
        print(json.dumps(summary, indent=2))
        return 0

    baseline = LongRAGBaseline(config=baseline_cfg, work_dir=index_dir)
    baseline.build_index(docs)

    runner = E2ERunner(
        baseline=baseline,
        reader=reader,
        main_prompt=main_prompt,
        extraction_prompt=extraction_prompt,
        demos=demos,
        max_context_chars=args.max_context_chars,
        dataset_name=dataset_name,
    )
    results = runner.run_batch(queries)

    manifest_path = out_dir / "manifest.json"
    runner.write_manifest(
        results,
        manifest_path,
        extra_meta={
            "docs_path": str(Path(args.docs).resolve()),
            "queries_path": str(Path(args.queries).resolve()),
            "baseline_config_path": str(Path(args.baseline_config).resolve()),
            "reader_config_path": str(Path(args.reader_config).resolve()),
            "main_prompt_name": main_prompt.name,
            "extraction_prompt_name": extraction_prompt.name,
            "dataset_name": dataset_name,
            "seed": args.seed,
            "num_documents": len(docs),
        },
    )

    answers_path = out_dir / "answers.jsonl"
    with answers_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "query_id": r.query_id,
                        "query": r.query,
                        "long_answer": r.long_answer,
                        "short_answer": r.short_answer,
                        "duration_ms": r.duration_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "status": "ok",
                "num_queries": len(results),
                "out_dir": str(out_dir),
                "manifest": str(manifest_path),
                "answers": str(answers_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())