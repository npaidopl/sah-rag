# ruff: noqa: E402
"""SAH-RAG M0-L3 prepare β€” Colab embedding input bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "outputs" / "external_sah_rag"
COLAB_ROOT = OUT_ROOT / "_colab_v1"
BUNDLE_VERSION = "v1"
RUNNER_NAME = "colab_sah_rag_m0_embed_chunks.py"

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
    chunks_dir = OUT_ROOT / ds / "chunks-v1"
    chunks_path = chunks_dir / "chunks.jsonl"
    manifest_path = chunks_dir / "manifest.json"
    if not chunks_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing chunks for {ds}: {chunks_dir}")

    bundle_path = COLAB_ROOT / f"sah_rag_colab_input_{ds}_{BUNDLE_VERSION}.tar.gz"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        bundle_path.unlink()

    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(chunks_path, arcname="input/chunks.jsonl")
        tar.add(manifest_path, arcname="input/chunks_manifest.json")
        tar.add(runner_src, arcname=RUNNER_NAME)

    return {
        "dataset_id": ds,
        "bundle_path": str(bundle_path.relative_to(REPO)).replace("\\", "/"),
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size,
        "chunks_jsonl_sha256": sha256_file(chunks_path),
    }


def write_readme(p: Path) -> None:
    p.write_text(
        "# SAH-RAG M0 Colab embedding run\n\n"
        "Upload all `sah_rag_colab_input_*_v1.tar.gz` bundles to Colab,\n"
        "then run:\n\n"
        "```python\n"
        "!pip -q install FlagEmbedding==1.4.2\n"
        "!python colab_sah_rag_m0_embed_chunks.py --all --out_dir /content/myrag_output\n"
        "```\n\n"
        "Outputs packed as `sah_rag_colab_output_<dataset>_v1.tar.gz`\n"
        "(dense + sparse, FP16).\n",
        encoding="utf-8",
    )


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

    write_readme(COLAB_ROOT / "README.md")
    manifest = {
        "artifact_type": "sah_rag_m0_colab_prepare_manifest",
        "artifact_version": "1.0",
        "execution_id": "sah-rag-m0-l3-prepare-v1",
        "status": "READY_FOR_COLAB",
        "bundle_version": BUNDLE_VERSION,
        "datasets": entries,
        "wall_clock_seconds": round(time.perf_counter() - t0, 3),
    }
    write_json_atomic(COLAB_ROOT / "prepare_manifest.json", manifest)
    print()
    print(f"M0-L3 PREPARE COMPLETE in {manifest['wall_clock_seconds']}s")
    total = sum(e["bundle_bytes"] for e in entries)
    print(f"  total bundle bytes: {total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())