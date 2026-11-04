"""Per-dataset query embedding cache. Dense + sparse, one file per dataset.

Layout: WORK_ROOT/query_embeddings/<dataset>/
  queries.jsonl   — one line per query: {query_id, dense_sha256_prefix}
  dense.npy       — float32 [N, 1024], row order matches queries.jsonl
  sparse.jsonl    — one line per query: {query_id, indices, values}
  manifest.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from ablation.config import WORK_ROOT


QUERY_EMBEDDINGS_DIR = WORK_ROOT / "query_embeddings"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class QueryEmbeddingCache:
    def __init__(self, dataset: str):
        base = QUERY_EMBEDDINGS_DIR / dataset
        self.dataset = dataset
        self.root = base
        self.queries_path = base / "queries.jsonl"
        self.dense_path = base / "dense.npy"
        self.sparse_path = base / "sparse.jsonl"
        self.manifest_path = base / "manifest.json"

    def exists(self) -> bool:
        return all(p.is_file() for p in [self.queries_path, self.dense_path,
                                          self.sparse_path, self.manifest_path])

    def save(self, query_ids: list[str], dense: np.ndarray,
             sparse: list[dict[int, float]], config: dict) -> None:
        if dense.shape != (len(query_ids), 1024):
            raise ValueError(f"Expected ({len(query_ids)}, 1024), got {dense.shape}")
        self.root.mkdir(parents=True, exist_ok=True)
        np.save(self.dense_path, dense)
        with self.queries_path.open("w", encoding="utf-8") as f:
            for qid in query_ids:
                f.write(json.dumps({"query_id": qid}) + "\n")
        with self.sparse_path.open("w", encoding="utf-8") as f:
            for qid, sp in zip(query_ids, sparse):
                ks = sorted(sp.keys())
                f.write(json.dumps({"query_id": qid, "indices": ks,
                                    "values": [sp[k] for k in ks]}) + "\n")
        self.manifest_path.write_text(json.dumps({
            "dataset": self.dataset, "count": len(query_ids),
            "dense_sha256": _sha256_file(self.dense_path),
            "sparse_sha256": _sha256_file(self.sparse_path),
            "config": config,
        }, indent=2), encoding="utf-8")

    def load(self) -> dict[str, tuple[np.ndarray, dict[int, float]]]:
        if not self.exists():
            raise FileNotFoundError(f"No query cache at {self.root}")
        dense = np.load(self.dense_path)
        query_ids = []
        with self.queries_path.open("r", encoding="utf-8") as f:
            for line in f:
                query_ids.append(json.loads(line)["query_id"])
        sparse = []
        with self.sparse_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                sparse.append(dict(zip(rec["indices"], rec["values"])))
        if dense.shape[0] != len(query_ids):
            raise ValueError("dense/query_id length mismatch")
        return {qid: (dense[i], sparse[i]) for i, qid in enumerate(query_ids)}