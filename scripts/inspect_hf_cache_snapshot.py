"""Inspect the HuggingFace cache for a specific model snapshot.

Read-only. Never triggers a download. Never loads model weights.
Reports file listing (including symlinks), blob inventory, refs, and
which config files are present. Reusable for any HF-hosted model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _hash_short(path: Path, n: int = 16) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _list_dir_robust(root: Path) -> list[dict]:
    entries: list[dict] = []
    if not root.exists():
        return entries
    for entry in sorted(root.iterdir()):
        is_link = entry.is_symlink()
        try:
            target = os.readlink(entry) if is_link else None
        except OSError:
            target = None
        size = None
        try:
            if entry.is_file():
                size = entry.stat().st_size
            elif entry.is_dir():
                size = sum(
                    f.stat().st_size
                    for f in entry.rglob("*")
                    if f.is_file()
                )
        except OSError:
            size = None
        entries.append({
            "name": entry.name,
            "is_symlink": is_link,
            "symlink_target": target,
            "is_file": entry.is_file(),
            "is_dir": entry.is_dir(),
            "size_bytes": size,
        })
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a HF cache snapshot read-only."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    snapshot_dir_name = "models--" + args.model_id.replace("/", "--")
    model_root = args.cache_root / snapshot_dir_name

    report: dict = {
        "artifact_type": "hf_cache_inspection",
        "artifact_version": "1.0",
        "model_id": args.model_id,
        "revision": args.revision,
        "cache_root": str(args.cache_root),
        "model_root": str(model_root),
        "model_root_exists": model_root.exists(),
    }

    if not model_root.exists():
        print(f"MODEL_ROOT_MISSING: {model_root}")
    else:
        # Top-level entries
        report["top_level"] = _list_dir_robust(model_root)

        # refs
        refs_dir = model_root / "refs"
        report["refs"] = {}
        if refs_dir.exists():
            for ref_file in refs_dir.iterdir():
                if ref_file.is_file():
                    report["refs"][ref_file.name] = ref_file.read_text(
                        encoding="utf-8"
                    ).strip()

        # blobs
        blobs_dir = model_root / "blobs"
        report["blobs"] = _list_dir_robust(blobs_dir)

        # Snapshot of the exact revision
        snapshot_root = model_root / "snapshots"
        report["snapshots_top_level"] = _list_dir_robust(snapshot_root)

        exact = snapshot_root / args.revision
        report["exact_snapshot_dir"] = str(exact)
        report["exact_snapshot_exists"] = exact.exists()
        if exact.exists():
            report["exact_snapshot_files"] = _list_dir_robust(exact)

            # Look for sharded weights and single-file weights
            all_names = [
                e["name"] for e in report["exact_snapshot_files"]
            ]
            report["has_single_safetensors"] = (
                "model.safetensors" in all_names
            )
            report["has_pytorch_bin"] = "pytorch_model.bin" in all_names
            report["has_index_json"] = (
                "model.safetensors.index.json" in all_names
                or "pytorch_model.bin.index.json" in all_names
            )
            report["has_config"] = "config.json" in all_names
            report["has_tokenizer_config"] = (
                "tokenizer_config.json" in all_names
            )
            report["has_sentencepiece"] = (
                "sentencepiece.bpe.model" in all_names
                or "spiece.model" in all_names
            )
            report["has_special_tokens_map"] = (
                "special_tokens_map.json" in all_names
            )

            # Sharded safetensors presence
            shards = [
                n for n in all_names
                if n.startswith("model-") and n.endswith(".safetensors")
            ]
            report["safetensors_shards"] = sorted(shards)

            # Total size of snapshot dir accounting for symlinks
            total = 0
            counted = 0
            for entry in exact.iterdir():
                try:
                    if entry.is_symlink():
                        real = entry.resolve()
                        if real.exists() and real.is_file():
                            total += real.stat().st_size
                            counted += 1
                    elif entry.is_file():
                        total += entry.stat().st_size
                        counted += 1
                except OSError:
                    pass
            report["snapshot_total_bytes_with_symlinks_resolved"] = total
            report["snapshot_file_count"] = counted

            # Verdict
            has_full_single = report["has_single_safetensors"]
            has_full_sharded = (
                report["has_index_json"] and len(shards) > 0
            )
            has_full_bin = report["has_pytorch_bin"]
            report["weights_complete"] = bool(
                has_full_single or has_full_sharded or has_full_bin
            )
            report["minimum_size_bytes_for_bge_m3_fp32"] = 2_000_000_000
            report["size_looks_plausible"] = total > 1_500_000_000

    # Write report
    out = args.out
    if out.exists():
        print(f"REFUSING_OVERWRITE: {out}")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
        newline=chr(10),
    )
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"WROTE  {out}  sha256={h}")
    print()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
