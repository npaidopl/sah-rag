"""BM25 indexer (rank_bm25). Persisted as JSON (tokenized corpus)."""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ablation.config import INDEXES_DIR


_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0370-\u03FF\u1F00-\u1FFF]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Indexer:
    def __init__(self, variant: str, dataset: str, root: Path | None = None):
        self.variant = variant
        self.dataset = dataset
        base = (root or INDEXES_DIR) / variant / dataset
        self.root = base
        self.tokens_path = base / "tokens.jsonl"
        self.manifest_path = base / "manifest.json"

    def exists(self) -> bool:
        return self.tokens_path.is_file() and self.manifest_path.is_file()

    def build_and_save(self, chunk_ids: list[str], texts: list[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have equal length")
        self.root.mkdir(parents=True, exist_ok=True)
        with self.tokens_path.open("w", encoding="utf-8") as f:
            for cid, txt in zip(chunk_ids, texts):
                f.write(json.dumps({"chunk_id": cid, "tokens": tokenize(txt)}) + "\n")
        self.manifest_path.write_text(json.dumps({
            "variant": self.variant, "dataset": self.dataset, "count": len(chunk_ids),
        }, indent=2), encoding="utf-8")

    def load(self):
        """Returns (chunk_ids, BM25Okapi instance)."""
        if not self.exists():
            raise FileNotFoundError(f"No BM25 index at {self.root}")
        chunk_ids, corpus = [], []
        with self.tokens_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                chunk_ids.append(rec["chunk_id"])
                corpus.append(rec["tokens"])
        return chunk_ids, BM25Okapi(corpus)