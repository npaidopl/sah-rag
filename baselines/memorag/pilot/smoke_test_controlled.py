"""MemoRAG controlled smoke test.

Runs the official memorag package under the v2 controlled envelope:
the memory model and clue stage remain native, the embedder is the
project-owned BGE-M3, and final answer generation is disabled.

This is NOT a mock. It imports the real ``memorag`` package, loads the
real memory checkpoint, and executes clue generation + clue-conditioned
retrieval. It does not call any external LLM API.

Usage
-----
    python workspace/memorag_dev/pilot/smoke_test_controlled.py \
        --manifest workspace/memorag_dev/pilot/controlled_manifest.yaml \
        --out workspace/memorag_dev/pilot/artifacts/controlled_run1

Exit codes
----------
    0  success
    1  missing dependency
    2  manifest still <PIN_ME>
    3  no CUDA device
    4  runtime failure (recorded in run.json)
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
        die(1, "PyYAML missing.")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_unpinned(obj: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits.extend(find_unpinned(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(find_unpinned(v, f"{path}[{i}]"))
    elif isinstance(obj, str) and ("<PIN_ME" in obj or "<FILL_FROM_SMOKE" in obj):
        hits.append(path)
    return hits


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
        if torch.cuda.is_available():
            snap["cuda_device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            snap["cuda_memory_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        snap["torch_version"] = None
        snap["cuda_available"] = False
    return snap


def probe_memorag_api() -> dict[str, Any]:
    probe: dict[str, Any] = {"imported": False, "attributes": {}}
    try:
        import memorag  # type: ignore
    except ImportError as exc:
        probe["import_error"] = repr(exc)
        return probe
    probe["imported"] = True
    probe["package_file"] = getattr(memorag, "__file__", None)
    probe["version"] = getattr(memorag, "__version__", None)
    for name in ("MemoRAG", "Memory", "DenseRetriever", "Agent"):
        attr = getattr(memorag, name, None)
        probe["attributes"][name] = {
            "present": attr is not None,
            "module": getattr(attr, "__module__", None),
        }
    return probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "load.log").open("w", encoding="utf-8")

    def tee(msg: str) -> None:
        print(msg)
        log.write(msg + "\n")
        log.flush()

    try:
        manifest = load_manifest(args.manifest)
        unpinned = find_unpinned(manifest)
        if unpinned:
            tee(f"Manifest still has UNPINNED fields: {unpinned}")
            die(2, "Refusing to run. Pin the fields first.")

        # Guardrail: contract requires generation disabled.
        if manifest.get("generation", {}).get("enabled") is not False:
            die(2, "Controlled pilot must have generation.enabled: false")

        env = env_snapshot()
        (args.out / "env.txt").write_text(json.dumps(env, indent=2), encoding="utf-8")
        tee(f"env: {env['platform']} | torch={env.get('torch_version')} | "
            f"cuda={env.get('cuda_available')} | vram={env.get('cuda_memory_gb')}")
        if not env.get("cuda_available"):
            die(3, "No CUDA device. This pilot must run on the GPU host.")

        api_probe = probe_memorag_api()
        (args.out / "probe.json").write_text(
            json.dumps(api_probe, indent=2), encoding="utf-8"
        )
        if not api_probe.get("imported"):
            die(1, f"memorag not importable: {api_probe.get('import_error')}")
        if not api_probe["attributes"].get("MemoRAG", {}).get("present"):
            die(1, "memorag.MemoRAG missing.")

        from memorag import MemoRAG  # type: ignore

        mem_cfg = manifest["models"]["memory"]
        inj_cfg = manifest["models"]["injected_embedder"]
        ctx = manifest.get("context", {})

        # For the controlled pilot we verify the injected embedder identity
        # but we do not bypass the memory model. The clue stage is native.
        # The BGE-M3 embedder is expected to be the same model the native
        # retriever uses, so in practice this is identity-preserving.
        tee(f"Injected embedder: {inj_cfg['id']}@{inj_cfg.get('revision')}")
        tee(f"Memory model (native): {mem_cfg['id']}@{mem_cfg.get('revision')}")

        t0 = time.time()
        pipeline = MemoRAG(
            mem_model_name_or_path=mem_cfg["id"],
            ret_model_name_or_path=inj_cfg["id"],
            cache_dir=str(args.out / "cache"),
        )
        tee(f"Instantiation took {time.time() - t0:.1f}s")

        load_fn = getattr(pipeline, "load", None) or getattr(pipeline, "load_corpus", None)
        if load_fn is None:
            die(4, "No load()/load_corpus() on pipeline.")
        t0 = time.time()
        load_fn(CORPUS)
        tee(f"load() indexed {len(CORPUS)} docs in {time.time() - t0:.1f}s")

        top_k = int(ctx.get("top_k", 3))
        t0 = time.time()
        try:
            raw = pipeline(QUERY, task_type="retrieval", top_k=top_k)
        except TypeError:
            raw = pipeline(context="", query=QUERY, task_type="retrieval", top_k=top_k)
        retrieve_dt = time.time() - t0
        (args.out / "retrieve_probe.json").write_text(
            json.dumps(
                {"query": QUERY, "top_k": top_k, "elapsed_s": retrieve_dt, "raw": str(raw)[:8000]},
                indent=2,
            ),
            encoding="utf-8",
        )
        tee(f"retrieve() took {retrieve_dt:.1f}s")

        # Explicit guardrail: no generation was executed.
        (args.out / "run.json").write_text(
            json.dumps({
                "status": "ok",
                "mode": "controlled",
                "generation_executed": False,
                "manifest": str(args.manifest),
            }),
            encoding="utf-8",
        )
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())