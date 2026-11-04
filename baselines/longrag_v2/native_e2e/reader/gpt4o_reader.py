"""GPT-4o reader (paper-default reader for LongRAG).

Paper alignment (from baseline_audit/longrag_native_reader_config.yaml):
    model         = gpt-4o
    temperature   = 0
    max_tokens    = 1000
    seed          = 12345
    retry         = 3
    delay_seconds = 5
    post_process  = regex on "answer:" prefix

Endpoint modes:
    - Azure OpenAI (paper default): set AZURE_OPENAI_ENDPOINT,
      AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_KEY (or OPENAI_API_KEY).
    - Standard OpenAI (fallback): set OPENAI_API_KEY only.

The reader auto-detects the endpoint type at first ``generate()`` call.

Determinism note: even with ``temperature=0`` and ``seed``, hosted LLMs
are not bit-reproducible. Cache or rerun when bit-identical comparison
matters.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# Paper post-processing: extract text after "answer:" (case-insensitive).
# Verbatim from utils/gpt_inference.py at pinned commit 778454e.
_POST_PROCESS_RE = re.compile(r"(?i)(?<=\banswer:\s).*")


class GPT4oReader:
    """OpenAI / Azure OpenAI reader for the native LongRAG pipeline."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        seed: int = 12345,
        max_tokens: int = 1000,
        retry: int = 3,
        delay_seconds: float = 5.0,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._seed = seed
        self._max_tokens = max_tokens
        self._retry = retry
        self._delay_seconds = delay_seconds
        self._azure_endpoint = azure_endpoint
        self._azure_api_version = azure_api_version
        self._client: Any = None
        self._is_azure = False

    def _load(self) -> None:
        if self._client is not None:
            return

        endpoint = self._azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = (
            self._azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        )

        if endpoint and api_version:
            api_key = (
                os.environ.get("AZURE_OPENAI_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            if not api_key:
                raise RuntimeError(
                    "Azure OpenAI mode requires AZURE_OPENAI_KEY "
                    "(or OPENAI_API_KEY). "
                    "Native end-to-end requires a reader key."
                )
            try:
                from openai import AzureOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "GPT4oReader (Azure) requires the `openai` package "
                    "(see requirements.txt)."
                ) from exc
            self._client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            self._is_azure = True
            logger.info(
                "Initialized GPT-4o reader (Azure): %s @ %s",
                self._model, endpoint,
            )
        else:
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
            self._is_azure = False
            logger.info(
                "Initialized GPT-4o reader (standard): %s", self._model
            )

    @property
    def name(self) -> str:
        return f"openai::{self._model}"

    @property
    def is_azure(self) -> bool:
        return self._is_azure

    @staticmethod
    def post_process(text: str) -> str:
        """Extract the substring after ``answer:`` if present.

        Verbatim behaviour from utils/gpt_inference.py at pinned commit.
        If no match, returns the input unchanged.
        """
        if not text:
            return text
        m = _POST_PROCESS_RE.search(text)
        return m.group(0) if m else text

    def generate(self, prompt: str) -> str:
        self._load()
        last_exc: Exception | None = None
        for attempt in range(self._retry):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    seed=self._seed,
                    max_tokens=self._max_tokens,
                )
                raw = (response.choices[0].message.content or "").strip()
                return self.post_process(raw)
            except Exception as exc:
                last_exc = exc
                if attempt < self._retry - 1:
                    time.sleep(self._delay_seconds)
        raise RuntimeError(
            f"GPT-4o reader failed after {self._retry} attempts: {last_exc}"
        ) from last_exc