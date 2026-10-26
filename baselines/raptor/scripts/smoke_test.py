"""Manual smoke test for RAPTOR baseline (transformers-only embedders).

Modes:
  dry        -- mock embedder + mock LLM (fast, no downloads)
  native     -- real BGE-M3 + real local LLM (per native_fidelity)
  paper      -- PAPER_EXACT_NATIVE (requires OPENAI_API_KEY)
  controlled -- real BGE-M3 + real LLM injected (mimics orchestrator)

Usage:
  python baselines/raptor/scripts/smoke_test.py --mode dry
  python baselines/raptor/scripts/smoke_test.py --mode native ^
      --embedder-path BAAI/bge-m3 ^
      --llm-path Qwen/Qwen2.5-1.5B-Instruct
  python baselines/raptor/scripts/smoke_test.py --mode controlled ^
      --embedder-path BAAI/bge-m3 ^
      --llm-path Qwen/Qwen2.5-1.5B-Instruct
  python baselines/raptor/scripts/smoke_test.py --mode paper
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.base import BaselineConfig, RunMode
from baselines.raptor.adapter import (
    RaptorBaseline,
    _BGEM3Embedder,
    _HFSummarizer,
    _SBERTMeanPoolEmbedder,
    _OpenAISummarizer,
)
from sah_rag.canonical.models import CanonicalDocument


SMOKE_DOCS = [
    "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet.",
    "Machine learning models require training data, compute, and evaluation to be useful. RAG combines retrieval with generation.",
    "Paris is the capital of France. It is known for the Eiffel Tower and the Louvre Museum. The Seine river flows through the city.",
    "Python is a high-level programming language. It emphasizes code readability and is widely used in data science.",
    "The mitochondria is the powerhouse of the cell. It produces ATP through oxidative phosphorylation.",
    "Climate change is driven by greenhouse gas emissions. Rising temperatures affect ecosystems globally.",
    "Shakespeare wrote many plays including Hamlet, Macbeth, and Romeo and Juliet. These are studied worldwide.",
    "The Milky Way is a barred spiral galaxy. Our solar system resides in one of its spiral arms.",
    "Coffee is brewed from roasted coffee beans. It contains caffeine, which is a stimulant.",
    "The Riemann hypothesis is a famous unsolved problem in mathematics about the distribution of prime numbers.",
]


class _DryEmbedder:
    name = "dry-embedder"
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
    def encode_documents(self, texts):
        import numpy as np
        rng = np.random.default_rng(42)
        return [rng.standard_normal(self.dim).tolist() for _ in texts]
    def encode_query(self, text):
        import numpy as np
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        return rng.standard_normal(self.dim).tolist()


class _DryLLM:
    name = "dry-llm"
    def generate(self, prompt, **kwargs):
        return f"SUMMARY: {prompt[:60].replace(chr(10), ' ')}"


def _build_real_embedder(path_or_id):
    class _RealEmbedder:
        name = "real-embedder-bge-m3"
        def __init__(self):
            self._inner = _BGEM3Embedder(path_or_id)
        def encode_documents(self, texts):
            return self._inner.encode(texts)
        def encode_query(self, text):
            return self._inner.encode_query(text)
    return _RealEmbedder()


def _build_real_llm(path_or_id):
    class _RealLLM:
        name = "real-llm-hf"
        def __init__(self):
            self._inner = _HFSummarizer(path_or_id)
        def generate(self, prompt, **kwargs):
            return self._inner.summarize(prompt, max_tokens=kwargs.get("max_tokens", 200))
    return _RealLLM()


def _make_docs():
    return [
        CanonicalDocument(
            doc_id=f"doc_{i:03d}", dataset="smoke",
            text_raw=text, text_normalized=text,
        )
        for i, text in enumerate(SMOKE_DOCS, start=1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry", "native", "paper", "controlled"], default="dry")
    parser.add_argument("--embedder-path", default=None)
    parser.add_argument("--llm-path", default=None)
    parser.add_argument("--work-dir", default="baselines/raptor/data/smoke_work")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    docs = _make_docs()
    print(f"[1] Loaded {len(docs)} documents")

    extra = {"chunk_size": 50, "max_levels": 2, "cluster_size": 3, "use_umap": False}

    if args.mode == "dry":
        cfg = BaselineConfig(name="raptor", mode=RunMode.CONTROLLED,
                             top_k=args.top_k, extra=extra)
        baseline = RaptorBaseline(
            config=cfg, work_dir=Path(args.work_dir),
            embedder=_DryEmbedder(), llm=_DryLLM(),
        )
        print("[2] DRY mode")
    elif args.mode == "native":
        if not args.embedder_path or not args.llm_path:
            print("ERROR: --embedder-path and --llm-path required")
            return 1
        extra["native_fidelity"] = "AUTHOR_SUPPORTED_NATIVE"
        extra["embedder_path"] = args.embedder_path
        extra["llm_path"] = args.llm_path
        cfg = BaselineConfig(name="raptor", mode=RunMode.NATIVE,
                             top_k=args.top_k, extra=extra)
        print(f"[2] NATIVE (AUTHOR_SUPPORTED): embedder={args.embedder_path}, llm={args.llm_path}")
        baseline = RaptorBaseline(config=cfg, work_dir=Path(args.work_dir))
    elif args.mode == "paper":
        extra["native_fidelity"] = "PAPER_EXACT_NATIVE"
        cfg = BaselineConfig(name="raptor", mode=RunMode.NATIVE,
                             top_k=args.top_k, extra=extra)
        print("[2] NATIVE (PAPER_EXACT): sbert + gpt-3.5-turbo (requires OPENAI_API_KEY)")
        baseline = RaptorBaseline(config=cfg, work_dir=Path(args.work_dir))
    else:  # controlled
        if not args.embedder_path or not args.llm_path:
            print("ERROR: --embedder-path and --llm-path required")
            return 1
        cfg = BaselineConfig(name="raptor", mode=RunMode.CONTROLLED,
                             top_k=args.top_k, extra=extra)
        print(f"[2] CONTROLLED: embedder={args.embedder_path}, llm={args.llm_path}")
        baseline = RaptorBaseline(
            config=cfg, work_dir=Path(args.work_dir),
            embedder=_build_real_embedder(args.embedder_path),
            llm=_build_real_llm(args.llm_path),
        )

    t0 = time.time()
    baseline.build_index(docs)
    print(f"[3] build_index OK in {time.time() - t0:.2f}s")

    tree_file = Path(args.work_dir) / "raptor_tree.jsonl"
    levels: dict[int, int] = {}
    with tree_file.open() as f:
        for line in f:
            node = json.loads(line)
            levels[node["level"]] = levels.get(node["level"], 0) + 1
    print("[4] Tree: " + ", ".join(f"L{k}={v}" for k, v in sorted(levels.items())))
    if max(levels) < 1:
        print("ERROR: tree has only level 0 -- NOT_RAPTOR_COMPLETE")
        return 1

    for q in ["capital of France", "RAG systems"]:
        t0 = time.time()
        results = baseline.retrieve(q, top_k=args.top_k)
        dt = time.time() - t0
        print(f"[5] Query: {q!r} -> {len(results)} results in {dt:.2f}s")
        for r in results:
            preview = r.text[:70].replace(chr(10), " ")
            print(f"     rank={r.rank} doc={r.doc_id} L={r.metadata.get('level')} "
                  f"score={r.score:.4f}  {preview}...")
        if not results:
            print("ERROR: empty results"); return 1
        if [r.rank for r in results] != list(range(1, len(results) + 1)):
            print("ERROR: ranks not contiguous"); return 1

    print("\n[OK] Smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())