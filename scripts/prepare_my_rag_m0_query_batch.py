# ruff: noqa: E402
"""My_RAG M0-L5 prepare — query bundles for Colab."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "outputs" / "external_my_rag"
QUERIES_DIR = REPO / "outputs" / "_shared_query_projections"
COLAB_ROOT = OUT_ROOT / "_colab_queries_v1"
RUNNER_NAME = "colab_my_rag_m0_embed_queries.py"
DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_json_atomic(p: Path, d: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def build_bundle(ds: str, runner_src: Path) -> dict[str, Any]:
    q_path = QUERIES_DIR / f"queries-{ds}.jsonl"
    if not q_path.exists():
        raise FileNotFoundError(f"Missing queries: {q_path}")
    bundle_path = COLAB_ROOT / f"my_rag_query_input_{ds}_v1.tar.gz"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        bundle_path.unlink()

    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(q_path, arcname="input/queries.jsonl")
        tar.add(runner_src, arcname=RUNNER_NAME)

    return {
        "dataset_id": ds,
        "bundle_path": str(bundle_path.relative_to(REPO)).replace("\\", "/"),
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size,
        "queries_jsonl_sha256": sha256_file(q_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()
    t0 = time.perf_counter()
    COLAB_ROOT.mkdir(parents=True, exist_ok=True)
    runner_src = REPO / "scripts" / RUNNER_NAME
    if not runner_src.exists():
        raise FileNotFoundError(f"Missing runner: {runner_src}")

    entries = []
    for ds in args.datasets:
        print(f"[prepare] {ds} ...")
        entries.append(build_bundle(ds, runner_src))

    write_json_atomic(COLAB_ROOT / "prepare_manifest.json", {
        "artifact_type": "my_rag_m0_query_prepare_manifest",
        "artifact_version": "1.0",
        "execution_id": "my-rag-m0-l5-prepare-v1",
        "status": "READY_FOR_COLAB",
        "datasets": entries,
        "wall_clock_seconds": round(time.perf_counter() - t0, 3),
    })
    total = sum(e["bundle_bytes"] for e in entries)
    print(f"\nM0-L5 PREPARE COMPLETE ({total:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())