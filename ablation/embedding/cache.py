"""Per-variant per-dataset embedding cache.

Layout: <EMBEDDINGS_DIR>/<variant>/<dataset>/
  dense.npy      — float32 [N, 1024]
  chunks.jsonl   — N lines: {"chunk_id":..., "indices":[...], "values":[...]}
  manifest.json  — count, sha256, config
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from ablation.config import EMBEDDINGS_DIR


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class EmbeddingCache:
    def __init__(self, variant: str, dataset: str, root: Path | None = None):
        self.variant = variant
        self.dataset = dataset
        base = (root or EMBEDDINGS_DIR) / variant / dataset
        self.root = base
        self.dense_path = base / "dense.npy"
        self.chunks_path = base / "chunks.jsonl"
        self.manifest_path = base / "manifest.json"

    def exists(self) -> bool:
        return (self.dense_path.is_file()
                and self.chunks_path.is_file()
                and self.manifest_path.is_file())

    def save(self, chunk_ids: list[str], dense: np.ndarray,
             sparse: list[dict[int, float]], config: dict) -> None:
        n = len(chunk_ids)
        if dense.shape[0] != n or len(sparse) != n:
            raise ValueError(f"Length mismatch: ids={n} dense={dense.shape} sparse={len(sparse)}")
        if dense.shape[1] != 1024:
            raise ValueError(f"Expected 1024 dims, got {dense.shape[1]}")

        self.root.mkdir(parents=True, exist_ok=True)
        np.save(self.dense_path, dense)

        with self.chunks_path.open("w", encoding="utf-8") as f:
            for cid, sp in zip(chunk_ids, sparse):
                ks = sorted(sp.keys())
                f.write(json.dumps({
                    "chunk_id": cid,
                    "indices": ks,
                    "values": [sp[k] for k in ks],
                }) + "\n")

        manifest = {
            "variant": self.variant,
            "dataset": self.dataset,
            "count": n,
            "dense_dim": int(dense.shape[1]),
            "dense_dtype": str(dense.dtype),
            "dense_sha256": _sha256_file(self.dense_path),
            "chunks_sha256": _sha256_file(self.chunks_path),
            "config": config,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load(self) -> tuple[list[str], np.ndarray, list[dict[int, float]]]:
        if not self.exists():
            raise FileNotFoundError(f"No cache at {self.root}")
        dense = np.load(self.dense_path)
        chunk_ids: list[str] = []
        sparse: list[dict[int, float]] = []
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                chunk_ids.append(rec["chunk_id"])
                sparse.append(dict(zip(rec["indices"], rec["values"])))
        return chunk_ids, dense, sparse