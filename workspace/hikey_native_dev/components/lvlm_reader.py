"""LVLM reader (paper Section 4.1, F.2, F.3).

Default: Qwen2.5-VL-7B-Instruct loaded locally (no API key needed).
Optional: API mode via environment variable HIKEY_LVLM_API_KEY.
If neither is available, ``generate`` raises a clear RuntimeError but
the retrieval pipeline still works.

The reader is OUT OF SCOPE for the v2 IR-only phase. This module exists
so the native end-to-end pipeline can be completed later without any
additional scaffolding work.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
API_KEY_ENV = "HIKEY_LVLM_API_KEY"
API_BASE_ENV = "HIKEY_LVLM_API_BASE"


QA_PROMPT_TEMPLATE = """[System Instruction]
You are an AI assistant that answers questions by analyzing the provided documents.
Write an accurate answer to [Question] using only the [Context] given below.
- Do not answer using information that is not present in the context.
- When referring to tables or figures, explicitly mention their IDs (e.g., Figure 3).
- If you cannot be confident, output "I do not have enough information to answer."

[Question]
{question}

[Context]
{context}

[Answer]
"""


class LVLMReader:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        api_mode: bool = False,
        mock: bool = False,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.api_mode = api_mode
        self.mock = mock
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.mock:
            return
        if self.api_mode:
            self._api_key = os.environ.get(API_KEY_ENV)
            self._api_base = os.environ.get(API_BASE_ENV)
            return
        try:
            from transformers import (  # type: ignore
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
            )
            import torch  # type: ignore
        except ImportError as e:
            logger.warning("transformers not available: %s", e)
            return
        try:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            self._processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception as e:
            logger.warning("Failed to load LVLM %s: %s", self.model_name, e)

    def generate(
        self,
        question: str,
        context: str,
        images: list[bytes] | None = None,
    ) -> str:
        self._load()
        prompt = QA_PROMPT_TEMPLATE.format(question=question, context=context)
        if self.mock:
            return "[MOCK_ANSWER] " + (question[:60] or "empty") + " ..."
        if self.api_mode:
            return self._api_generate(prompt, images)
        if self._model is None or self._processor is None:
            raise RuntimeError(
                "LVLM not loaded. Either (a) install transformers + download "
                "Qwen2.5-VL-7B-Instruct, or (b) set " + API_KEY_ENV + " to use API mode."
            )
        return self._local_generate(prompt, images)

    def _local_generate(self, prompt: str, images: list[bytes] | None) -> str:
        from PIL import Image  # type: ignore
        import io, torch  # type: ignore
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in (images or [])]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=pil_images or None, return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = out[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]

    def _api_generate(self, prompt: str, images: list[bytes] | None) -> str:
        if not getattr(self, "_api_key", None):
            raise RuntimeError(
                f"API mode enabled but {API_KEY_ENV} is not set. "
                "Set it in your environment or switch to local mode."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError("openai package required for API mode.") from e
        client = OpenAI(api_key=self._api_key, base_url=self._api_base or None)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""