"""Shared embedder implementations for controlled-mode injection.

Implements the ``baselines.base.Embedder`` Protocol. Models are lazily
loaded and cached as a module-level singleton. No top-level heavy imports
(torch, transformers) — they are only imported on first use.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_ROOT = r"D:\rag_work\hf_cache\models--BAAI--bge-m3"


def _resolve_model_path() -> Path:
    """Resolve the BGE-M3 model path.

    Priority:
      1. RAG_EMBEDDER_ROOT env var (if set)
      2. Default D:/rag_work/hf_cache/models--BAAI--bge-m3

    If the resolved directory has a ``snapshots/`` subdirectory, the first
    snapshot is used. Otherwise the directory itself is returned.
    """
    root_str = os.environ.get("RAG_EMBEDDER_ROOT", _DEFAULT_ROOT)
    root = Path(root_str)
    if not root.exists():
        raise RuntimeError(
            f"BGE-M3 model directory not found: {root}. "
            f"Set RAG_EMBEDDER_ROOT or place the model at {_DEFAULT_ROOT}."
        )
    snap = root / "snapshots"
    if snap.exists():
        candidates = [p for p in snap.iterdir() if p.is_dir()]
        if candidates:
            return sorted(candidates)[0]
    return root


class BGEM3Embedder:
    """BGE-M3 dense embedder (CLS pooling + L2 normalization).

    Matches the pooling used by the RAPTOR reference adapter so that
    controlled-mode comparisons across baselines share identical vector
    semantics.
    """

    _MAX_LENGTH = 512
    _BATCH_SIZE = 16

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        max_length: int | None = None,
        allow_download: bool = False,
    ) -> None:
        self._model_path = Path(model_path) if model_path else _resolve_model_path()
        self._max_length = int(max_length or self._MAX_LENGTH)
        self._allow_download = allow_download
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    # ---- lazy loading ----

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BGEM3Embedder requires torch and transformers. "
                f"Cause: {exc}"
            ) from exc

        local_only = not self._allow_download
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_path), local_files_only=local_only
            )
            self._model = AutoModel.from_pretrained(
                str(self._model_path), local_files_only=local_only
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load BGE-M3 from {self._model_path}. "
                f"Set RAG_EMBEDDER_ROOT or allow_download=True. Cause: {exc}"
            ) from exc

        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.to("cuda")
        self._torch = torch
        log.info("BGEM3Embedder loaded from %s", self._model_path)

    # ---- Embedder Protocol ----

    @property
    def name(self) -> str:
        return "bge-m3"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        if not texts:
            return []
        torch = self._torch
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model(**inputs)
            dense = torch.nn.functional.normalize(
                out.last_hidden_state[:, 0], p=2, dim=1
            )
            all_vecs.extend(dense.cpu().tolist())
        return all_vecs

    def encode_query(self, text: str) -> list[float]:
        return self.encode_documents([text])[0]