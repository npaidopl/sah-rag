"""GPT-4o reader (paper-default reader for LongRAG).

Requires ``OPENAI_API_KEY`` in the environment. The paper also reports a
Gemini-1.5-Pro configuration (see ``gemini_reader.py``); both are
author-supported.

Determinism note: even with ``temperature=0`` and ``seed``, hosted LLMs are
not bit-reproducible. Cache or rerun when bit-identical comparison matters.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class GPT4oReader:
    """OpenAI Chat Completions reader for the native LongRAG pipeline."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._seed = seed
        self._max_tokens = max_tokens
        self._client: Any = None

    def _load(self) -> None:
        if self._client is not None:
            return

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Native end-to-end requires a reader key."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "GPT4oReader requires the `openai` package "
                "(see requirements.txt)."
            ) from exc

        self._client = OpenAI(api_key=api_key)
        logger.info("Initialized GPT-4o reader: %s", self._model)

    @property
    def name(self) -> str:
        return f"openai::{self._model}"

    def generate(self, prompt: str) -> str:
        self._load()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            seed=self._seed,
            max_tokens=self._max_tokens,
        )
        return (response.choices[0].message.content or "").strip()