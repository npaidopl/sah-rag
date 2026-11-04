"""BGE-M3 hybrid embedder wrapper (dense + sparse). CPU/GPU agnostic."""
import os
import time

import numpy as np

from ablation.hf_cache import resolve_snapshot


class BGEM3Embedder:
    def __init__(self, max_length: int = 512, use_fp16: bool = False,
                 device: str = "auto", batch_size: int = 32):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from FlagEmbedding import BGEM3FlagModel

        self.snapshot_path = resolve_snapshot("BAAI/bge-m3")
        self.dense_dim = 1024
        self.max_length = max_length
        self.batch_size = batch_size

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        t0 = time.time()
        self.model = BGEM3FlagModel(
            str(self.snapshot_path),
            use_fp16=(use_fp16 and device == "cuda"),
            device=device,
        )
        self.load_time_s = time.time() - t0

    def encode_hybrid(self, texts: list[str]):
        """Returns (dense: np.ndarray [N,1024] float32, sparse: list[dict[int,float]])."""
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        sparse = [{int(k): float(v) for k, v in d.items()}
                  for d in out["lexical_weights"]]
        return dense, sparse