"""Smoke test: load BGE-M3 from the pinned local snapshot and encode 2 strings.

Fails fast if the snapshot is missing or FlagEmbedding is not installed.
Does not write any artifacts and does not touch the network.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Force offline before importing the encoder.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "D:/rag_work/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

SNAPSHOT_PATH = Path(
    "D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)

from rag_system.baselines.longrag_bge_m3 import (  # noqa: E402
    BGE_M3_DENSE_DIMENSION,
    BgeM3DenseEncoder,
)


def main() -> None:
    print("===== SMOKE: LONGRAG SYNTHETIC ACCEPTANCE =====")
    if not SNAPSHOT_PATH.exists():
        print(f"[FAIL] snapshot missing: {SNAPSHOT_PATH}")
        sys.exit(1)
    print(f"[OK] snapshot: {SNAPSHOT_PATH}")

    print("[OK] building encoder (this loads the model once)...")
    encoder = BgeM3DenseEncoder(
        snapshot_path=SNAPSHOT_PATH,
        device="cpu",
        use_fp16=False,
    )

    texts = [
        "The Amber Observatory records atmospheric patterns.",
        "The Cedar Archive stores transferred documents in numbered shelves.",
    ]
    print("[OK] encoding 2 probe strings...")
    vectors = encoder.encode_dense(texts, batch_size=2, max_length=512)

    if len(vectors) != 2:
        print(f"[FAIL] expected 2 vectors, got {len(vectors)}")
        sys.exit(1)
    for i, v in enumerate(vectors):
        if len(v) != BGE_M3_DENSE_DIMENSION:
            print(
                f"[FAIL] vector {i}: dim={len(v)} != {BGE_M3_DENSE_DIMENSION}"
            )
            sys.exit(1)

    print(f"[OK] dim={BGE_M3_DENSE_DIMENSION}")
    print(f"[OK] v0[0:3]={vectors[0][:3]}")
    print(f"[OK] v1[0:3]={vectors[1][:3]}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()