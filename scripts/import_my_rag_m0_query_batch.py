# ruff: noqa: E402
"""My_RAG M0-L5 import — query embeddings back to local."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "outputs" / "external_my_rag"
COLAB_ROOT = OUT_ROOT / "_colab_queries_v1"
QUERY_EMB_ROOT = OUT_ROOT / "_query_embeddings"
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--out_dir", default=str(COLAB_ROOT))
    args = ap.parse_args()

    t0 = time.perf_counter()
    QUERY_EMB_ROOT.mkdir(parents=True, exist_ok=True)
    work = COLAB_ROOT / "_import_work"
    work.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)

    results = []
    for ds in args.datasets:
        cand = out_dir / f"my_rag_query_output_{ds}_v1.tar.gz"
        if not cand.exists():
            print(f"[skip] {cand.name}")
            continue
        print(f"[import] {ds} ...")
        extract = work / ds
        extract.mkdir(parents=True, exist_ok=True)
        with tarfile.open(cand, "r:gz") as tar:
            tar.extractall(extract)
        recs = extract / "output" / "query_embeddings.jsonl"
        man = extract / "output" / "manifest.json"
        if not recs.exists() or not man.exists():
            print("  missing files; skip")
            continue

        dst_recs = QUERY_EMB_ROOT / f"{ds}.jsonl"
        dst_man = QUERY_EMB_ROOT / f"{ds}.manifest.json"
        dst_recs.write_bytes(recs.read_bytes())
        dst_man.write_text(man.read_text(encoding="utf-8"), encoding="utf-8")

        m = json.loads(dst_man.read_text(encoding="utf-8"))
        actual_sha = sha256_file(dst_recs)
        if actual_sha != m["query_embeddings_sha256"]:
            print("  SHA mismatch")
            continue
        print(f"  {m['record_count']} records -> {dst_recs.name}")
        results.append({"dataset_id": ds, "record_count": m["record_count"]})

    write_json_atomic(COLAB_ROOT / "import_manifest.json", {
        "artifact_type": "my_rag_m0_query_import_manifest",
        "artifact_version": "1.0",
        "execution_id": "my-rag-m0-l5-import-v1",
        "status": "SUCCEEDED",
        "datasets": results,
        "wall_clock_seconds": round(time.perf_counter() - t0, 3),
    })
    print()
    for r in results:
        print(f"  {r['dataset_id']:18s} records={r['record_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())