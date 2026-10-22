"""Provider-neutral contracts for answer-generation runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sah_rag.generation.prompts import (
    GenerationPrompt,
)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One provider-neutral generation request."""

    prompt: GenerationPrompt

    model_id: str
    maximum_output_tokens: int

    temperature: float = 0.0
    seed: int | None = None

    response_format: str = "text"


@dataclass(frozen=True, slots=True)
class GenerationRuntimeResponse:
    """Normalized response returned by a generation runtime."""

    answer_text: str

    model_id: str
    model_revision: str | None
    runtime_id: str

    input_token_count: int
    output_token_count: int

    finish_reason: str | None

    request_id: str | None = None

    metadata: dict[str, object] | None = None


@runtime_checkable
class GenerationRuntimeAdapter(Protocol):
    """Contract implemented by every generation provider."""

    @property
    def runtime_id(self) -> str:
        """Return the stable runtime identifier."""

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationRuntimeResponse:
        """Generate one normalized runtime response."""
