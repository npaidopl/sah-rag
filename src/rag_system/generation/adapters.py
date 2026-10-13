"""Generation-runtime adapter implementations."""

from __future__ import annotations

from rag_system.generation.runtime import (
    GenerationRequest,
    GenerationRuntimeResponse,
)


class DeterministicDryRunAdapter:
    """Return a controlled response without model inference."""

    def __init__(
        self,
        *,
        answer_text: str,
    ) -> None:
        """Initialize the dry-run adapter."""

        if not answer_text.strip():
            raise ValueError(
                "Dry-run answer text cannot be empty."
            )

        self._answer_text = answer_text

    @property
    def runtime_id(self) -> str:
        """Return the dry-run runtime identity."""

        return "deterministic-dry-run"

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationRuntimeResponse:
        """Return the configured deterministic response."""

        if request.maximum_output_tokens < 1:
            raise ValueError(
                "maximum_output_tokens must be positive."
            )

        if not 0.0 <= request.temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0 and 2."
            )

        return GenerationRuntimeResponse(
            answer_text=self._answer_text,
            model_id=request.model_id,
            model_revision=None,
            runtime_id=self.runtime_id,
            input_token_count=(
                request.prompt.rendered_prompt_tokens
            ),
            output_token_count=len(
                self._answer_text.split()
            ),
            finish_reason="dry_run_complete",
            request_id=None,
            metadata={
                "model_inference_performed": False,
                "synthetic": True,
                "response_format": (
                    request.response_format
                ),
                "seed": request.seed,
            },
        )
