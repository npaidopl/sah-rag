"""Locate a HuggingFace-style or direct model directory on disk.

READ-ONLY: does not modify, delete, or move anything. Does not load
model weights. Does not touch the network. Reports candidate model
roots with config/weights presence and total size.

Usage:
  python scripts/locate_local_model.py \
      --model-id BAAI/bge-m3 \
      --roots "D:\\rag_work" "D:\\" \
      --max-depth 6 \
      --out baseline_audit/model_location_probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


HF_DIR_NAME = "models--{org}--{name}"
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__",
             "target", "build", "dist", ".cache_old"}
CONFIG_MARKERS = {"config.json", "sentence_bert_config.json"}
WEIGHT_MARKERS = {"model.safetensors", "pytorch_model.bin"}


def _dir_size(root: Path, max_files: int = 5000) -> tuple[int, int]:
    total = 0
    count = 0
    try:
        for f in root.rglob("*"):
            if count >= max_files:
                break
            try:
                if f.is_file():
                    total += f.stat().st_size
                    count += 1
            except OSError:
                continue
    except OSError:
        pass
    return total, count


def _list_files(root: Path, limit: int = 200) -> list[dict]:
    out: list[dict] = []
    try:
        for f in sorted(root.iterdir()):
            if len(out) >= limit:
                break
            try:
                is_link = f.is_symlink()
                target = None
                if is_link:
                    try:
                        target = str(f.resolve())
                    except OSError:
                        target = None
                size = None
                if f.is_file():
                    size = f.stat().st_size
                out.append({
                    "name": f.name,
                    "is_dir": f.is_dir(),
                    "is_file": f.is_file(),
                    "is_symlink": is_link,
                    "symlink_target": target,
                    "size_bytes": size,
                })
            except OSError:
                continue
    except OSError:
        pass
    return out


def _classify(root: Path) -> dict:
    """Return {'hf_cache': bool, 'direct_model': bool, ...} for one dir."""
    names = set()
    try:
        names = {p.name for p in root.iterdir()}
    except OSError:
        pass
    has_config = "config.json" in names
    has_sbert = "sentence_bert_config.json" in names
    has_single_weight = bool(WEIGHT_MARKERS & names)
    has_shards = any(
        n.startswith("model-") and n.endswith(".safetensors") for n in names
    )
    has_index = (
        "model.safetensors.index.json" in names
        or "pytorch_model.bin.index.json" in names
    )
    is_hf_cache_root = (
        (root / "blobs").is_dir() and (root / "snapshots").is_dir()
        and (root / "refs").is_dir()
    )
    return {
        "has_config": has_config,
        "has_sbert_config": has_sbert,
        "has_single_weight_file": has_single_weight,
        "has_sharded_weights": has_shards,
        "has_weight_index": has_index,
        "is_hf_cache_root": is_hf_cache_root,
        "weights_present": bool(
            has_single_weight or (has_shards and has_index)
        ),
    }


def _walk_and_match(root: Path, model_dir_name: str, max_depth: int) -> list[dict]:
    hits: list[dict] = []
    root = root.resolve()
    if not root.exists():
        return hits
    root_depth = len(root.parts)

    for dirpath, dirnames, _filenames in __import__("os").walk(
        str(root), topdown=True
    ):
        current = Path(dirpath)
        depth = len(current.parts) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        name = current.name

        matches_name = name == model_dir_name
        try:
            cls = _classify(current)
        except Exception:
            cls = {}

        if matches_name or (
            cls.get("weights_present") and (cls.get("has_config") or cls.get("has_sbert_config"))
        ):
            size_bytes, file_count = _dir_size(current)
            hit = {
                "path": str(current),
                "matched_by": (
                    "hf_cache_dir_name" if matches_name
                    else "direct_model_layout"
                ),
                "depth": depth,
                "size_bytes": size_bytes,
                "file_count_sampled": file_count,
                "classify": cls,
                "first_files": _list_files(current, limit=60),
            }
            # If it is an HF cache root, inspect refs + snapshots
            if cls.get("is_hf_cache_root"):
                refs_dir = current / "refs"
                refs: dict = {}
                if refs_dir.exists():
                    for rf in refs_dir.iterdir():
                        if rf.is_file():
                            try:
                                refs[rf.name] = rf.read_text(
                                    encoding="utf-8"
                                ).strip()
                            except OSError:
                                pass
                hit["refs"] = refs

                snaps = current / "snapshots"
                snap_list: list[dict] = []
                if snaps.exists():
                    for s in sorted(snaps.iterdir()):
                        if not s.is_dir():
                            continue
                        try:
                            sz, cnt = _dir_size(s, max_files=2000)
                        except Exception:
                            sz, cnt = (0, 0)
                        snap_list.append({
                            "name": s.name,
                            "size_bytes": sz,
                            "file_count_sampled": cnt,
                            "files": _list_files(s, limit=60),
                        })
                hit["snapshots"] = snap_list

            hits.append(hit)

    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Locate a model directory on disk (read-only)."
    )
    parser.add_argument("--model-id", required=True,
                        help="e.g. BAAI/bge-m3")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    org, _, name = args.model_id.partition("/")
    model_dir_name = HF_DIR_NAME.format(org=org.replace("/", "--"), name=name)

    report: dict = {
        "artifact_type": "local_model_location_probe",
        "artifact_version": "1.0",
        "model_id": args.model_id,
        "model_dir_name": model_dir_name,
        "roots": [str(r) for r in args.roots],
        "max_depth": args.max_depth,
        "hits": [],
    }

    for r in args.roots:
        r_resolved = r.resolve() if r.exists() else r
        hits = _walk_and_match(r_resolved, model_dir_name, args.max_depth)
        report["hits"].extend(hits)

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
    print(f"total_hits: {len(report['hits'])}")
    for i, hit in enumerate(report["hits"], 1):
        print(f"  hit #{i}: {hit['path']}")
        print(f"    matched_by: {hit['matched_by']}")
        print(f"    size_bytes: {hit['size_bytes']}")
        print(f"    weights_present: {hit['classify'].get('weights_present')}")
        if "refs" in hit:
            print(f"    refs: {hit['refs']}")
        if "snapshots" in hit:
            for snap in hit["snapshots"]:
                print(
                    f"      snapshot {snap['name']}: "
                    f"{snap['size_bytes']} bytes, "
                    f"{snap['file_count_sampled']} files"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
