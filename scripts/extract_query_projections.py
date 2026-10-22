# ruff: noqa: E402
"""Extract query projections (query_id + query_text) for embedding.

Deliberately drops all ground truth fields. Output is safe to upload
to any external compute environment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from sah_rag.canonical.loaders import iter_canonical_examples

OUT = REPO_ROOT / "outputs" / "_shared_query_projections"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")

for ds in DATASETS:
    rows: list[dict[str, str]] = []
    for ex in iter_canonical_examples(ds):
        rows.append({
            "query_id": ex.query.query_id,
            "query_text": ex.query.question,
        })
    path = OUT / f"queries-{ds}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(
                json.dumps(
                    r, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                )
            )
            fh.write("\n")
    print(f"{ds}: {len(rows)} queries -> {path} ({path.stat().st_size} bytes)")

print("\nDONE")