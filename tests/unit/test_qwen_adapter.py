"""Tests for sah_rag.generation.qwen_adapter.QwenLLMAdapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from sah_rag.generation.prompts import GenerationPrompt
from sah_rag.generation.qwen_adapter import QwenLLMAdapter
from sah_rag.generation.runtime import (
    GenerationRequest,
    GenerationRuntimeResponse,
)


# --------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------


class _MockQwenLLM:
    """Records calls and returns configured text."""

    def __init__(self, answer: str = "The answer is [C1].") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return self.answer


def _make_prompt() -> GenerationPrompt:
    return GenerationPrompt(
        prompt_id="prompt-test-001",
        context_pack_id="context-test-001",
        query_id="q-1",
        dataset_id="clerk",
        template_id="clerk_cited_qa",
        template_version="1.0.0",
        tokenizer_id="word-counter-v1",
        source_context_tokens=10,
        rendered_prompt_tokens=20,
        maximum_prompt_tokens=100,
        reserved_answer_tokens=20,
        system_instruction="Answer using the sources.",
        rendered_context="[C1]\nsource_text:\nThe law says X.",
        user_prompt="Question:\nWhat does the law say?\n\nSources:\n[C1]\n...",
        citation_ids=("[C1]",),
        source_chunk_ids=("chunk-1",),
        source_document_ids=("doc-1",),
    )


def _make_request(prompt: GenerationPrompt | None = None) -> GenerationRequest:
    return GenerationRequest(
        prompt=prompt or _make_prompt(),
        model_id="Qwen/Qwen2.5-14B-Instruct",
        maximum_output_tokens=100,
        temperature=0.0,
        seed=None,
        response_format="text",
    )


# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------


def test_adapter_uses_injected_llm():
    """Adapter should call generate() with correct prompt assembly."""

    mock = _MockQwenLLM(answer="Yes, [C1].")
    adapter = QwenLLMAdapter(llm=mock)

    response = adapter.generate(_make_request())

    assert isinstance(response, GenerationRuntimeResponse)
    assert response.answer_text == "Yes, [C1]."
    assert response.runtime_id == "qwen-llm-adapter"
    assert response.model_id == "Qwen/Qwen2.5-14B-Instruct"
    assert response.finish_reason == "stop"
    assert response.metadata["model_inference_performed"] is True

    assert len(mock.calls) == 1
    call = mock.calls[0]
    # Full prompt = system_instruction + "\n\n" + user_prompt
    assert "Answer using the sources." in call["prompt"]
    assert "What does the law say?" in call["prompt"]
    assert call["kwargs"]["max_tokens"] == 100
    assert call["kwargs"]["temperature"] == 0.0


def test_adapter_custom_runtime_id():
    mock = _MockQwenLLM()
    adapter = QwenLLMAdapter(llm=mock, runtime_id="custom-runtime-42")
    response = adapter.generate(_make_request())
    assert response.runtime_id == "custom-runtime-42"


def test_adapter_empty_runtime_id_raises():
    with pytest.raises(ValueError, match="runtime_id cannot be empty"):
        QwenLLMAdapter(llm=_MockQwenLLM(), runtime_id="   ")


def test_adapter_non_string_return_raises():
    class BadLLM:
        def generate(self, prompt: str, **kwargs: Any) -> Any:
            return 42  # not a string

    adapter = QwenLLMAdapter(llm=BadLLM())
    with pytest.raises(RuntimeError, match="expected str"):
        adapter.generate(_make_request())


def test_adapter_token_accounting():
    """input_token_count mirrors prompt.rendered_prompt_tokens,
    output_token_count = word count of answer."""

    mock = _MockQwenLLM(answer="one two three four five")
    adapter = QwenLLMAdapter(llm=mock)
    response = adapter.generate(_make_request())

    assert response.input_token_count == 20  # from _make_prompt
    assert response.output_token_count == 5  # word count


def test_adapter_property_runtime_id():
    adapter = QwenLLMAdapter(llm=_MockQwenLLM())
    assert adapter.runtime_id == "qwen-llm-adapter"