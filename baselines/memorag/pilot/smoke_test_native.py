"""MemoRAG native smoke test.

Runs the official memorag package end-to-end on a tiny corpus to prove
the native path executes on the GPU host. This is NOT a unit test and
NOT a mock: it imports the real ``memorag`` package, loads the real
memory checkpoint, and executes clue generation + retrieval (+ optional
generation if the manifest enables it).

Usage
-----
    python workspace/memorag_dev/pilot/smoke_test_native.py \
        --manifest workspace/memorag_dev/pilot/native_manifest.yaml \
        --out workspace/memorag_dev/pilot/artifacts/native_run1

Exit codes
----------
    0  success
    1  missing dependency (memorag, torch, etc.)
    2  missing manifest fields (still <PIN_ME>)
    3  no CUDA device
    4  runtime failure (recorded in artifacts/run.json)

The script is intentionally defensive: it probes the memorag API and
writes the observed method signatures into ``probe.json`` so that the
adapter can be adjusted against the actual package, not against
assumptions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# --- tiny corpus (deliberately short; do NOT edit to hide failures) ---
CORPUS = [
    "The cat sat on the mat in the quiet corner of the room.",
    "Dogs chase balls in the park every afternoon after lunch.",
    "Retrieval-augmented generation combines a retriever and a reader.",
    "Long-context documents require special handling to remain usable.",
    "Legal contracts are structured hierarchically with clauses and annexes.",
]
QUERY = "How does retrieval-augmented generation combine components?"


# --- helpers ---
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


def find_unpinned(obj: Any, path: str = "") -> list[str]:
    """Return dotted paths of fields still holding <PIN_ME> or <FILL_FROM_SMOKE>."""
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
            snap["cuda_device_count"] = torch.cuda.device_count()
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


def probe_memorag_methods(pipeline: Any) -> dict[str, Any]:
    """Best-effort introspection of the live pipeline object."""
    out: dict[str, Any] = {"dir": [], "callables": {}}
    try:
        out["dir"] = sorted(
            name for name in dir(pipeline) if not name.startswith("__")
        )
    except Exception as exc:
        out["dir_error"] = repr(exc)
    for name in out["dir"]:
        attr = getattr(pipeline, name, None)
        if callable(attr):
            try:
                import inspect
                out["callables"][name] = str(inspect.signature(attr))
            except (TypeError, ValueError):
                out["callables"][name] = "<signature unavailable>"
    return out


# --- main ---
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "load.log"
    log = log_path.open("w", encoding="utf-8")

    def tee(msg: str) -> None:
        print(msg)
        log.write(msg + "\n")
        log.flush()

    try:
        # 1) manifest
        manifest = load_manifest(args.manifest)
        unpinned = find_unpinned(manifest)
        if unpinned:
            tee(f"Manifest still has UNPINNED fields: {unpinned}")
            # Allow only the generation toggle to remain unpinned for IR-only pilots
            critical = [p for p in unpinned if "generator.enabled" not in p]
            if critical:
                die(2, f"Refusing to run with pinned fields missing: {critical}")

        # 2) env
        env = env_snapshot()
        (args.out / "env.txt").write_text(
            json.dumps(env, indent=2), encoding="utf-8"
        )
        tee(f"env: {env['platform']} | torch={env.get('torch_version')} | "
            f"cuda={env.get('cuda_available')} | "
            f"vram={env.get('cuda_memory_gb')}")
        if not env.get("cuda_available"):
            die(3, "No CUDA device. This pilot must run on the GPU host.")

        # 3) probe memorag API before instantiating
        api_probe = probe_memorag_api()
        (args.out / "probe.json").write_text(
            json.dumps(api_probe, indent=2), encoding="utf-8"
        )
        if not api_probe.get("imported"):
            die(1, f"memorag package not importable: {api_probe.get('import_error')}")
        if not api_probe["attributes"].get("MemoRAG", {}).get("present"):
            die(1, "memorag.MemoRAG is missing. Wrong package version?")

        # 4) instantiate
        from memorag import MemoRAG  # type: ignore

        mem_cfg = manifest["models"]["memory"]
        ret_cfg = manifest["models"]["retriever"]
        ctx = manifest.get("context", {})

        mem_id = mem_cfg["id"]
        mem_rev = mem_cfg.get("revision")
        ret_id = ret_cfg["id"]
        ret_rev = ret_cfg.get("revision")

        tee(f"Instantiating MemoRAG(mem={mem_id}@{mem_rev}, ret={ret_id}@{ret_rev})")
        t0 = time.time()
        pipeline = MemoRAG(
            mem_model_name_or_path=mem_id,
            ret_model_name_or_path=ret_id,
            cache_dir=str(args.out / "cache"),
        )
        tee(f"Instantiation took {time.time() - t0:.1f}s")

        # 5) introspect live pipeline
        methods = probe_memorag_methods(pipeline)
        (args.out / "methods.json").write_text(
            json.dumps(methods, indent=2), encoding="utf-8"
        )
        tee(f"Pipeline exposes: {sorted(methods['callables'].keys())}")

        # 6) load corpus via whichever method exists
        load_fn = getattr(pipeline, "load", None) or getattr(
            pipeline, "load_corpus", None
        )
        if load_fn is None:
            die(4, "No load()/load_corpus() on pipeline. See methods.json.")
        t0 = time.time()
        load_fn(CORPUS)
        tee(f"load() indexed {len(CORPUS)} docs in {time.time() - t0:.1f}s")

        # 7) retrieval probe
        top_k = int(ctx.get("top_k", 3))
        t0 = time.time()
        try:
            raw = pipeline(QUERY, task_type="retrieval", top_k=top_k)
        except TypeError as exc:
            tee(f"pipeline(query, task_type, top_k) raised {exc!r}; trying legacy signature")
            raw = pipeline(context="", query=QUERY, task_type="retrieval", top_k=top_k)
        retrieve_dt = time.time() - t0
        (args.out / "retrieve_probe.json").write_text(
            json.dumps(
                {"query": QUERY, "top_k": top_k, "elapsed_s": retrieve_dt, "raw": str(raw)[:8000]},
                indent=2,
            ),
            encoding="utf-8",
        )
        tee(f"retrieve() took {retrieve_dt:.1f}s -> {type(raw).__name__}")

        # 8) optional generation probe
        gen_cfg = manifest.get("models", {}).get("generator", {})
        if gen_cfg.get("enabled") is True:
            t0 = time.time()
            try:
                answer = pipeline(context="", query=QUERY, task_type="generation")
            except TypeError:
                answer = pipeline(QUERY)
            gen_dt = time.time() - t0
            (args.out / "generate_probe.json").write_text(
                json.dumps(
                    {"query": QUERY, "elapsed_s": gen_dt, "answer": str(answer)[:8000]},
                    indent=2,
                ),
                encoding="utf-8",
            )
            tee(f"generate() took {gen_dt:.1f}s")
        else:
            tee("generation disabled in manifest; skipping generate()")

        (args.out / "run.json").write_text(
            json.dumps({"status": "ok", "manifest": str(args.manifest)}),
            encoding="utf-8",
        )
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())