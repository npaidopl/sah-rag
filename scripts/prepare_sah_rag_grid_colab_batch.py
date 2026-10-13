# ruff: noqa: E402
"""SAH-RAG M3-prep: package 36 grid variants into Colab bundles."""
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
COLAB_ROOT = OUT_ROOT / "_grid_colab_v1"
INPUT_DIR = COLAB_ROOT / "input"
RUNNER_NAME = "colab_sah_rag_grid_embed_chunks.py"
BUNDLE_VERSION = "v1"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")
GRID: tuple[tuple[int, int], ...] = (
    (256, 0), (256, 32), (256, 128),
    (512, 0), (512, 32), (512, 128),
    (1024, 0), (1024, 32), (1024, 128),
)


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


def chunk_dir(ds: str, cs: int, ov: int) -> Path:
    return OUT_ROOT / ds / "grid-chunks-v1" / f"c{cs}_o{ov}"


def bundle_name(ds: str, cs: int, ov: int) -> str:
    return f"sah_rag_grid_input_{ds}_c{cs}_o{ov}_{BUNDLE_VERSION}.tar.gz"


def build_bundle(
    ds: str, cs: int, ov: int, runner_src: Path,
) -> dict[str, Any]:
    chunks_path = chunk_dir(ds, cs, ov) / "chunks.jsonl"
    chunks_mf = chunk_dir(ds, cs, ov) / "manifest.json"
    if not chunks_path.exists() or not chunks_mf.exists():
        raise FileNotFoundError(f"Missing grid chunks: {chunk_dir(ds, cs, ov)}")

    mf = json.loads(chunks_mf.read_text(encoding="utf-8"))
    if mf["chunk_size"] != cs or mf["chunk_overlap"] != ov:
        raise RuntimeError(f"{ds} c{cs}_o{ov}: manifest mismatch")

    bundle_path = INPUT_DIR / bundle_name(ds, cs, ov)
    if bundle_path.exists():
        bundle_path.unlink()

    grid_meta = {
        "dataset_id": ds,
        "chunk_size": cs,
        "chunk_overlap": ov,
        "chunker_config_hash": mf["chunker_config_hash"],
        "chunk_count": mf["chunk_count"],
        "chunks_jsonl_sha256": mf["chunks_jsonl_sha256"],
    }

    # temp grid_meta.json
    tmp_meta = COLAB_ROOT / "_tmp_grid_meta.json"
    write_json_atomic(tmp_meta, grid_meta)

    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(chunks_path, arcname="input/chunks.jsonl")
        tar.add(chunks_mf, arcname="input/chunks_manifest.json")
        tar.add(tmp_meta, arcname="input/grid_meta.json")
        tar.add(runner_src, arcname=RUNNER_NAME)

    tmp_meta.unlink(missing_ok=True)

    return {
        "dataset_id": ds,
        "chunk_size": cs,
        "chunk_overlap": ov,
        "variant_id": f"c{cs}_o{ov}",
        "bundle_path": str(bundle_path.relative_to(REPO)).replace("\\", "/"),
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size,
        "chunks_jsonl_sha256": mf["chunks_jsonl_sha256"],
        "chunk_count": mf["chunk_count"],
        "chunker_config_hash": mf["chunker_config_hash"],
    }


def write_readme(p: Path) -> None:
    p.write_text(
        "# SAH-RAG M3 grid β€” Colab embedding run\n\n"
        "Upload every `sah_rag_grid_input_*.tar.gz` to Colab.\n\n"
        "```python\n"
        "!pip -q install FlagEmbedding==1.4.2\n"
        "!python colab_sah_rag_grid_embed_chunks.py --all "
        "--out_dir /content/grid_out\n"
        "```\n\n"
        "Runner saves each output bundle to Drive immediately.\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()

    t0 = time.perf_counter()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner_src = REPO / "scripts" / RUNNER_NAME
    if not runner_src.exists():
        raise FileNotFoundError(f"Missing runner: {runner_src}")

    entries: list[dict[str, Any]] = []
    for ds in args.datasets:
        for cs, ov in GRID:
            print(f"[prepare] {ds} c{cs}_o{ov} ...")
            entries.append(build_bundle(ds, cs, ov, runner_src))

    write_readme(COLAB_ROOT / "README.md")
    manifest = {
        "artifact_type": "sah_rag_grid_colab_prepare_manifest",
        "artifact_version": "1.0",
        "execution_id": "sah-rag-m3-prepare-v1",
        "status": "READY_FOR_COLAB",
        "bundle_version": BUNDLE_VERSION,
        "bundle_count": len(entries),
        "bundles": entries,
        "wall_clock_seconds": round(time.perf_counter() - t0, 3),
    }
    write_json_atomic(COLAB_ROOT / "prepare_manifest.json", manifest)

    total = sum(e["bundle_bytes"] for e in entries)
    total_chunks = sum(e["chunk_count"] for e in entries)
    print()
    print(f"M3-PREP COMPLETE in {manifest['wall_clock_seconds']}s")
    print(f"  bundles: {len(entries)}")
    print(f"  total bundle bytes: {total:,}")
    print(f"  total chunks: {total_chunks:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())