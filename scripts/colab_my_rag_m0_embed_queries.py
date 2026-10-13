"""My_RAG M0-L5 Colab query embedding runner (dense + sparse)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DENSE_DIMENSION = 1024
MAX_LENGTH = 4098
BATCH_SIZE = 32


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


def write_jsonl_atomic(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
            fh.write("\n")
    tmp.replace(p)


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_sparse(lex: dict[Any, Any]) -> tuple[list[int], list[float]]:
    items = [(int(k), float(v)) for k, v in lex.items() if float(v) != 0.0]
    items.sort(key=lambda p: p[0])
    return [i for i, _ in items], [v for _, v in items]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out_dir", default="/content/myrag_q_out")
    ap.add_argument("--work_dir", default="/content/myrag_q_work")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

    import torch
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from FlagEmbedding import BGEM3FlagModel
    print("Loading BGE-M3 hybrid FP16 ...")
    model = BGEM3FlagModel(MODEL_ID, revision=MODEL_REVISION,
                          use_fp16=True, device="cuda")

    if args.all:
        bundles = sorted(Path.cwd().glob("my_rag_query_input_*_v1.tar.gz"))
    else:
        bundles = [Path(b) for b in args.bundles]
    if not bundles:
        print("No bundles.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for b in bundles:
        print(f"\n=== {b.name} ===")
        t0 = time.time()
        extract = work_dir / b.stem
        extract.mkdir(parents=True, exist_ok=True)
        with tarfile.open(b, "r:gz") as tar:
            tar.extractall(extract)
        queries_path = extract / "input" / "queries.jsonl"
        if not queries_path.exists():
            print("  missing queries; skip")
            continue
        rows = read_jsonl(queries_path)
        print(f"  {len(rows)} queries")

        ds = b.name.replace("my_rag_query_input_", "").replace("_v1.tar.gz", "")
        cfg_hash = hashlib.sha256(
            f"{MODEL_ID}:{MODEL_REVISION}:hybrid:fp16:gpu:query:{MAX_LENGTH}".encode()
        ).hexdigest()

        texts = [r["query_text"] for r in rows]
        enc = model.encode(
            texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
            return_dense=True, return_sparse=True, return_colbert_vecs=False,
        )
        dense = enc["dense_vecs"]
        sparse = enc["lexical_weights"]

        out_rows = []
        for i, r in enumerate(rows):
            dv = [float(x) for x in dense[i]]
            if len(dv) != DENSE_DIMENSION:
                raise RuntimeError(f"dim {len(dv)} at {i}")
            sidx, sval = normalize_sparse(sparse[i])
            out_rows.append({
                "query_id": r["query_id"],
                "query_text_sha256": hashlib.sha256(
                    r["query_text"].encode("utf-8")
                ).hexdigest(),
                "dense_vector": dv,
                "sparse_vector": {"indices": sidx, "values": sval},
                "dense_dimension": DENSE_DIMENSION,
                "dense_normalized": True,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "embedding_config_hash": cfg_hash,
                "precision": "float16",
                "max_length": MAX_LENGTH,
                "device": "gpu",
            })

        # write output bundle
        stage = out_dir / f"stage_{ds}"
        stage.mkdir(parents=True, exist_ok=True)
        recs = stage / "query_embeddings.jsonl"
        man = stage / "manifest.json"
        write_jsonl_atomic(recs, out_rows)
        recs_sha = sha256_file(recs)
        write_json_atomic(man, {
            "artifact_type": "my_rag_m0_query_embeddings_manifest",
            "artifact_version": "1.0",
            "dataset_id": ds,
            "record_count": len(out_rows),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "embedding_config_hash": cfg_hash,
            "dense_dimension": DENSE_DIMENSION,
            "precision": "float16",
            "max_length": MAX_LENGTH,
            "query_embeddings_sha256": recs_sha,
        })

        out_bundle = out_dir / f"my_rag_query_output_{ds}_v1.tar.gz"
        if out_bundle.exists():
            out_bundle.unlink()
        with tarfile.open(out_bundle, "w:gz") as tar:
            tar.add(recs, arcname="output/query_embeddings.jsonl")
            tar.add(man, arcname="output/manifest.json")
        print(f"  -> {out_bundle.name} ({out_bundle.stat().st_size/1e6:.1f} MB)")

        summary.append({
            "dataset_id": ds,
            "record_count": len(out_rows),
            "out_bundle_sha256": sha256_file(out_bundle),
            "out_bundle_bytes": out_bundle.stat().st_size,
            "seconds": round(time.time() - t0, 1),
        })

    write_json_atomic(out_dir / "colab_summary.json", {
        "artifact_type": "my_rag_m0_query_colab_summary",
        "artifact_version": "1.0",
        "datasets": summary,
    })
    print("\nCOLAB QUERY RUN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())