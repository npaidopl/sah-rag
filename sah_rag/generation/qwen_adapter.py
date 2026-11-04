"""Generation-runtime adapter that wraps the local QwenLLM component."""

from __future__ import annotations

from typing import Any

from sah_rag.generation.runtime import (
    GenerationRequest,
    GenerationRuntimeResponse,
)


class QwenLLMAdapter:
    """GenerationRuntimeAdapter backed by sah_rag.components.llms.QwenLLM.

    This adapter is the "koumpoma" that turns SAH-RAG from a pure
    information-retrieval system into an end-to-end RAG system, as
    required by the paper.

    Design notes
    ------------
    * QwenLLM is imported lazily on first generate() call, so that
      importing this module is always cheap and does not require the
      Qwen weights to be on disk. This allows offline tests with a
      custom/mock LLM instance.
    * QwenLLM.generate() hardcodes max_length=2048 on the input
      tokenizer. Prompts larger than that are silently truncated.
      The pipeline defaults keep maximum_prompt_tokens well below this
      ceiling.
    * QwenLLM does not expose an EOS/finish-reason signal, so
      finish_reason is always reported as "stop".
    * The controlled fairness protocol mandates Qwen2.5-14B-Instruct.
      QwenLLM enforces this by default (strict_fairness=True).
    """

    def __init__(
        self,
        llm: Any | None = None,
        *,
        runtime_id: str = "qwen-llm-adapter",
    ) -> None:
        if not runtime_id.strip():
            raise ValueError("runtime_id cannot be empty.")

        self._llm = llm
        self._runtime_id = runtime_id

    @property
    def runtime_id(self) -> str:
        """Return the stable runtime identifier."""

        return self._runtime_id

    def _get_llm(self) -> Any:
        """Lazily instantiate QwenLLM on first use."""

        if self._llm is None:
            from sah_rag.components.llms import QwenLLM

            self._llm = QwenLLM()
        return self._llm

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationRuntimeResponse:
        """Generate one answer using QwenLLM."""

        prompt = request.prompt

        full_prompt = (
            f"{prompt.system_instruction}\n\n{prompt.user_prompt}"
        )

        llm = self._get_llm()

        answer_text = llm.generate(
            full_prompt,
            max_tokens=request.maximum_output_tokens,
            temperature=request.temperature,
        )

        if not isinstance(answer_text, str):
            raise RuntimeError(
                "QwenLLM.generate returned "
                f"{type(answer_text).__name__}, expected str."
            )

        return GenerationRuntimeResponse(
            answer_text=answer_text,
            model_id=request.model_id,
            model_revision=None,
            runtime_id=self.runtime_id,
            input_token_count=prompt.rendered_prompt_tokens,
            output_token_count=len(answer_text.split()),
            finish_reason="stop",
            request_id=None,
            metadata={
                "model_inference_performed": True,
                "temperature": request.temperature,
                "maximum_output_tokens": request.maximum_output_tokens,
                "full_prompt_char_length": len(full_prompt),
            },
        )