"""Gemini-1.5-Pro reader (paper-supported reader for LongRAG).

Requires ``GOOGLE_API_KEY`` in the environment. Author-supported but not the
paper default; GPT-4o achieves the highest EM in the paper comparison.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class GeminiReader:
    """Google Generative AI reader for the native LongRAG pipeline."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._client: Any = None

    def _load(self) -> None:
        if self._client is not None:
            return

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Native end-to-end requires a reader key."
            )

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "GeminiReader requires `google-generativeai` "
                "(see requirements.txt)."
            ) from exc

        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(self._model)
        logger.info("Initialized Gemini reader: %s", self._model)

    @property
    def name(self) -> str:
        return f"google::{self._model}"

    def generate(self, prompt: str) -> str:
        self._load()
        response = self._client.generate_content(
            prompt,
            generation_config={
                "temperature": self._temperature,
                "max_output_tokens": self._max_output_tokens,
            },
        )
        return (response.text or "").strip()