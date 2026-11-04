"""Resolve HF snapshot paths across two cache layouts (root and hub/)."""
from __future__ import annotations
from pathlib import Path
from ablation.config import HF_CACHE_ROOT

def resolve_snapshot(repo_id: str, revision: str | None = None) -> Path:
    """Return snapshot path containing model weights.

    Searches both layouts:
      D:\\rag_work\\hf_cache\\models--<org>--<name>\\snapshots\\<rev>\\
      D:\\rag_work\\hf_cache\\hub\\models--<org>--<name>\\snapshots\\<rev>\\
    Prefers snapshots with *.safetensors or pytorch_model.bin.
    """
    dirname = "models--" + repo_id.replace("/", "--")
    roots = [HF_CACHE_ROOT, HF_CACHE_ROOT / "hub"]
    candidates: list[Path] = []
    for root in roots:
        snap_root = root / dirname / "snapshots"
        if not snap_root.is_dir():
            continue
        if revision:
            p = snap_root / revision
            if p.is_dir():
                candidates.append(p)
        else:
            candidates.extend(sorted(snap_root.iterdir(),
                                     key=lambda p: p.stat().st_mtime,
                                     reverse=True))
    for c in candidates:
        if any(c.glob("*.safetensors")) or (c / "pytorch_model.bin").is_file():
            return c
    raise FileNotFoundError(f"No weights snapshot for {repo_id} (rev={revision}) under {HF_CACHE_ROOT}")