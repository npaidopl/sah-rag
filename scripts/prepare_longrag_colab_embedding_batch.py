# ruff: noqa: E402
"""LongRAG L3 - Prepare Colab-GPU embedding bundle.

For each dataset already materialized under
outputs/external_longrag/<dataset>/units-v1/, create one self-contained
tar.gz bundle plus a shared Colab runner script and a README.

The bundle contains only the units.jsonl + manifest + summary + checksums
required by the Colab runner. No model weights, no query text, no ground
truth. The runner embeds text only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "external_longrag"
COLAB_ROOT = OUTPUT_ROOT / "_colab_v1"

EXPECTED_DATASETS: tuple[str, ...] = (
    "ruler",
    "loong",
    "long_bench_v2",
    "lara",
    "greek_bar_bench",
    "cuad",
    "clerk",
)

BUNDLE_VERSION = "v1"
COLAB_RUNNER_NAME = "colab_longrag_embed_long_units.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        .encode("utf-8")
        + b"\n"
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(data))


def build_bundle(dataset_id: str, colab_runner_src: Path) -> dict[str, Any]:
    src_dir = OUTPUT_ROOT / dataset_id / "units-v1"
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing units-v1: {src_dir}")

    bundle_path = COLAB_ROOT / f"longrag_colab_input_{dataset_id}_{BUNDLE_VERSION}.tar.gz"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        bundle_path.unlink()

    files = ["units.jsonl", "manifest.json", "summary.json", "checksums.json"]
    for f in files:
        if not (src_dir / f).exists():
            raise FileNotFoundError(f"Missing {src_dir / f}")

    with tarfile.open(bundle_path, "w:gz") as tar:
        for f in files:
            tar.add(src_dir / f, arcname=f"input/{f}")
        tar.add(colab_runner_src, arcname=COLAB_RUNNER_NAME)

    bundle_sha = sha256_file(bundle_path)
    bundle_size = bundle_path.stat().st_size

    return {
        "dataset_id": dataset_id,
        "bundle_path": str(bundle_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "bundle_sha256": bundle_sha,
        "bundle_bytes": bundle_size,
        "units_jsonl_sha256": sha256_file(src_dir / "units.jsonl"),
    }


def write_colab_readme(path: Path) -> None:
    text = (
        "# LongRAG Colab embedding run\n"
        "\n"
        "Upload all `longrag_colab_input_*_v1.tar.gz` bundles to the\n"
        "Colab session, then run:\n"
        "\n"
        "```python\n"
        "!pip -q install FlagEmbedding==1.4.2\n"
        "!python colab_longrag_embed_long_units.py --all \\\n"
        "    --out_dir /content/longrag_colab_output\n"
        "```\n"
        "\n"
        "The runner will, for each bundle:\n"
        "1. Verify the input SHA-256 against the input manifest.\n"
        "2. Extract units.jsonl.\n"
        "3. Load BGE-M3 dense-only in FP16 on CUDA.\n"
        "4. Embed every long unit (dense, normalized).\n"
        "5. Write embedding_records.jsonl + embedding_manifest.json +\n"
        "   checksums.json.\n"
        "\n"
        "Outputs are packed as `longrag_colab_output_<dataset>_v1.tar.gz`.\n"
        "Download them and run the local import script.\n"
    )
    atomic_write_bytes(path, text.encode("utf-8"))


def run(*, datasets: tuple[str, ...], force: bool) -> int:
    t0 = time.perf_counter()
    COLAB_ROOT.mkdir(parents=True, exist_ok=True)

    colab_runner_src = REPO_ROOT / "scripts" / COLAB_RUNNER_NAME
    if not colab_runner_src.exists():
        raise FileNotFoundError(
            f"Missing Colab runner source: {colab_runner_src}"
        )

    entries: list[dict[str, Any]] = []
    for ds in datasets:
        print(f"[prepare] {ds} ...")
        entries.append(build_bundle(ds, colab_runner_src))

    write_colab_readme(COLAB_ROOT / "README.md")

    prepare_manifest: dict[str, Any] = {
        "artifact_type": "longrag_colab_embedding_prepare_manifest",
        "artifact_version": "1.0",
        "execution_id": "longrag-l3-prepare-v1",
        "status": "READY_FOR_COLAB",
        "datasets": entries,
        "bundle_version": BUNDLE_VERSION,
        "wall_clock_seconds": round(time.perf_counter() - t0, 3),
        "next_gate": "colab_gpu_embedding_run",
    }
    write_json_atomic(COLAB_ROOT / "prepare_manifest.json", prepare_manifest)

    print()
    print(f"L3 PREPARE COMPLETE in {prepare_manifest['wall_clock_seconds']}s")
    print(f"  {COLAB_ROOT}")
    total_bytes = sum(e["bundle_bytes"] for e in entries)
    print(f"  total bundle bytes: {total_bytes:,}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="*", default=list(EXPECTED_DATASETS),
        choices=list(EXPECTED_DATASETS),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(datasets=tuple(args.datasets), force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())