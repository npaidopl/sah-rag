"""Multimodal encoders (paper Section 4.1).

Text:  gte-Qwen2-7B-instruct  (via FlagEmbedding)
Visual: SigLIP (google/siglip-large-patch16-384) or MM-Embed

When ``mock=True``, encoders return deterministic hash vectors
(384-dim) so tests and CI run offline without model downloads.
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


MOCK_DIM = 384


def _mock_vec(text: str, dim: int = MOCK_DIM) -> list[float]:
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    return [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]


class MultiModalEncoder:
    def __init__(
        self,
        *,
        text_model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
        visual_model_name: str = "google/siglip-large-patch16-384",
        cache_dir: Path | str | None = None,
        mock: bool = False,
        device: str = "auto",
    ) -> None:
        self.text_model_name = text_model_name
        self.visual_model_name = visual_model_name
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.mock = mock
        self.device = device
        self._text_model: Any = None
        self._visual_model: Any = None
        self._visual_processor: Any = None

    def _load_text(self) -> Any:
        if self._text_model is not None:
            return self._text_model
        try:
            from FlagEmbedding import FlagModel  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "FlagEmbedding not installed. Install with: pip install FlagEmbedding"
            ) from e
        logger.info("Loading text encoder %s", self.text_model_name)
        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._text_model = FlagModel(
            self.text_model_name,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=True,
            **kwargs,
        )
        return self._text_model

    def _load_visual(self) -> tuple[Any, Any]:
        if self._visual_model is not None:
            return self._visual_model, self._visual_processor
        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "transformers not installed. Install with: pip install transformers torch"
            ) from e
        logger.info("Loading visual encoder %s", self.visual_model_name)
        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._visual_processor = AutoProcessor.from_pretrained(self.visual_model_name, **kwargs)
        self._visual_model = AutoModel.from_pretrained(self.visual_model_name, **kwargs)
        self._visual_model.eval()
        return self._visual_model, self._visual_processor

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if self.mock:
            return [_mock_vec(t) for t in texts]
        model = self._load_text()
        vecs = model.encode(texts, batch_size=8, max_length=8192)
        return [v.tolist() for v in vecs]

    def encode_image(self, image_bytes: bytes) -> list[float]:
        if self.mock:
            return _mock_vec(f"img:{len(image_bytes)}")
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        model, proc = self._load_visual()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        return out[0].cpu().numpy().tolist()