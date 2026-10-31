"""Local (HuggingFace transformers) reader for sanity checks only.

This reader is NOT for native-fidelity claims. It exists so the end-to-end
pipeline can be exercised on a laptop or on a GPU box without API keys.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LocalReader:
    """Wrap a HuggingFace causal LM as a Reader.

    Loading is lazy: the model is only loaded on the first ``generate()`` call,
    so importing this module is safe on machines without transformers/torch.

    Parameters
    ----------
    model_name
        HuggingFace model id. Defaults to a small instruct model suitable
        for sanity checks. Do not use large models here.
    max_new_tokens
        Generation budget.
    device
        Passed to ``transformers.pipeline(device_map=...)``. ``"auto"`` picks
        GPU if available, CPU otherwise.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 256,
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._device = device
        self._pipeline: Any = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "LocalReader requires `transformers` and `torch`. "
                "Install them or use GPT4oReader / GeminiReader."
            ) from exc
        logger.info("Loading local reader model: %s", self._model_name)
        self._pipeline = pipeline(
            "text-generation",
            model=self._model_name,
            device_map=self._device,
            torch_dtype="auto",
        )

    @property
    def name(self) -> str:
        return f"local::{self._model_name}"

    def generate(self, prompt: str) -> str:
        self._load()
        result = self._pipeline(
            prompt,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return (result[0].get("generated_text") or "").strip()