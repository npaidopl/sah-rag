"""MemoRAG native end-to-end run (no external API key required).

Loads the official memory checkpoint (Qwen2-7B), builds memory over a
tiny corpus, generates clues, retrieves, and generates an answer using
the SAME checkpoint as the generator. This is the official MemoRAG
default generator path and requires no OpenAI/Anthropic key.

CPU is supported. Expect several minutes per query; this is a proof of
execution, not a benchmark.

Usage:
    python run_native_e2e.py \
      --manifest pilot/native_manifest.yaml \
      --out pilot/artifacts/native_e2e_<timestamp>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CORPUS = [
    "The cat sat on the mat in the quiet corner of the room.",
    "Dogs chase balls in the park every afternoon after lunch.",
    "Retrieval-augmented generation combines a retriever and a reader.",
    "Long-context documents require special handling to remain usable.",
    "Legal contracts are structured hierarchically with clauses and annexes.",
]
QUERY = "How does retrieval-augmented generation combine components?"


def die(code: int, msg: str) -> "NoReturn":
    print(f"[FATAL {code}] {msg}", file=sys.stderr)
    sys.exit(code)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        die(1, "PyYAML missing. pip install pyyaml")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def env_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }
    try:
        import torch  # type: ignore
        snap["torch_version"] = torch.__version__
        snap["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        snap["torch_version"] = None
        snap["cuda_available"] = False
    return snap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "load.log").open("w", encoding="utf-8")

    def tee(msg: str) -> None:
        print(msg)
        log.write(msg + "\n")
        log.flush()

    try:
        manifest = load_manifest(args.manifest)
        mem_cfg = manifest["models"]["memory"]
        ret_cfg = manifest["models"]["retriever"]
        gen_cfg = manifest["models"].get("generator", {})

        if not gen_cfg.get("enabled"):
            die(2, "generator.enabled is false; use smoke_test_native.py for IR-only")

        env = env_snapshot()
        (args.out / "env.txt").write_text(json.dumps(env, indent=2), encoding="utf-8")
        tee(f"env: {env['platform']} | torch={env.get('torch_version')} | "
            f"cuda={env.get('cuda_available')}")

        try:
            import memorag  # type: ignore
        except ImportError:
            die(1, "memorag not installed. pip install memorag")

        from memorag import MemoRAG  # type: ignore

        (args.out / "probe.json").write_text(
            json.dumps(
                {
                    "package_file": getattr(memorag, "__file__", None),
                    "version": getattr(memorag, "__version__", None),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Local generation: memory model acts as generator (no API key).
        tee(f"Memory:   {mem_cfg['id']}@{mem_cfg.get('revision')}")
        tee(f"Retriever: {ret_cfg['id']}@{ret_cfg.get('revision')}")
        tee(f"Generator: {gen_cfg['id']}@{gen_cfg.get('revision')} (local)")

        t0 = time.time()
        pipeline = MemoRAG(
            mem_model_name_or_path=mem_cfg["id"],
            ret_model_name_or_path=ret_cfg["id"],
            cache_dir=str(args.out / "cache"),
        )
        tee(f"Instantiation: {time.time() - t0:.1f}s")

        load_fn = getattr(pipeline, "load", None) or getattr(pipeline, "load_corpus", None)
        if load_fn is None:
            die(4, "No load()/load_corpus() on pipeline")
        t0 = time.time()
        load_fn(CORPUS)
        tee(f"load(): {time.time() - t0:.1f}s ({len(CORPUS)} docs)")

        t0 = time.time()
        try:
            retrieve_out = pipeline(QUERY, task_type="retrieval", top_k=3)
        except TypeError:
            retrieve_out = pipeline(context="", query=QUERY, task_type="retrieval", top_k=3)
        (args.out / "retrieve_probe.json").write_text(
            json.dumps(
                {"query": QUERY, "elapsed_s": time.time() - t0,
                 "raw": str(retrieve_out)[:8000]},
                indent=2,
            ),
            encoding="utf-8",
        )
        tee(f"retrieve(): {time.time() - t0:.1f}s")

        t0 = time.time()
        try:
            answer = pipeline(context=str(retrieve_out), query=QUERY, task_type="generation")
        except TypeError:
            answer = pipeline(QUERY)
        (args.out / "generate_probe.json").write_text(
            json.dumps(
                {"query": QUERY, "elapsed_s": time.time() - t0,
                 "answer": str(answer)[:8000],
                 "generator_model": gen_cfg["id"]},
                indent=2,
            ),
            encoding="utf-8",
        )
        tee(f"generate(): {time.time() - t0:.1f}s")

        (args.out / "run.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "native_e2e": True,
                    "generator_model": gen_cfg["id"],
                    "generator_local": True,
                    "requires_api_key": False,
                    "manifest": str(args.manifest),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())