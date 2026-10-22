"""Typed generated-answer artifacts and citation validation."""

from __future__ import annotations

import re

from pydantic import Field, model_validator

from sah_rag.artifacts import (
    ArtifactBaseModel,
    sha256_fingerprint,
)
from sah_rag.generation.prompts import (
    GenerationPrompt,
)
from sah_rag.generation.runtime import (
    GenerationRequest,
    GenerationRuntimeResponse,
)

_CITATION_PATTERN = re.compile(
    r"\[C[1-9][0-9]*\]"
)


class AnswerCitationValidation(ArtifactBaseModel):
    """Citation-integrity report for one generated answer."""

    available_citation_ids: tuple[str, ...]
    used_citation_ids: tuple[str, ...]
    unknown_citation_ids: tuple[str, ...]
    unused_citation_ids: tuple[str, ...]

    citation_occurrence_count: int = Field(ge=0)
    has_citations: bool
    all_citations_known: bool
    valid: bool

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> AnswerCitationValidation:
        """Validate consistency of citation sets."""

        for values in (
            self.available_citation_ids,
            self.used_citation_ids,
            self.unknown_citation_ids,
            self.unused_citation_ids,
        ):
            if len(values) != len(set(values)):
                raise ValueError(
                    "Citation-report identifiers must be unique."
                )

        available = set(
            self.available_citation_ids
        )
        used = set(
            self.used_citation_ids
        )
        unknown = set(
            self.unknown_citation_ids
        )
        unused = set(
            self.unused_citation_ids
        )

        if unknown != used.difference(available):
            raise ValueError(
                "unknown_citation_ids are inconsistent."
            )

        if unused != available.difference(used):
            raise ValueError(
                "unused_citation_ids are inconsistent."
            )

        if self.has_citations != bool(
            self.used_citation_ids
        ):
            raise ValueError(
                "has_citations is inconsistent."
            )

        if self.all_citations_known != (
            len(self.unknown_citation_ids) == 0
        ):
            raise ValueError(
                "all_citations_known is inconsistent."
            )

        if self.valid != (
            self.has_citations
            and self.all_citations_known
        ):
            raise ValueError(
                "Citation validation status is inconsistent."
            )

        return self


class GeneratedAnswer(ArtifactBaseModel):
    """One generated answer with runtime provenance."""

    artifact_type: str = "generated_answer"
    artifact_version: str = "1.0"

    answer_id: str
    prompt_id: str
    context_pack_id: str

    query_id: str
    dataset_id: str

    answer_text: str

    generation_model_id: str
    generation_model_revision: str | None = None
    generation_runtime: str
    input_token_count: int = Field(ge=0)
    output_token_count: int = Field(ge=0)

    finish_reason: str | None = None

    refused: bool = False
    insufficient_context: bool = False

    citation_validation: (
        AnswerCitationValidation
    )

    generation_parameters: dict[
        str,
        object,
    ] = Field(default_factory=dict)

    metadata: dict[str, object] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_answer(
        self,
    ) -> GeneratedAnswer:
        """Validate answer identity and citation state."""

        required_values = (
            self.answer_id,
            self.prompt_id,
            self.context_pack_id,
            self.query_id,
            self.dataset_id,
            self.answer_text,
            self.generation_model_id,
            self.generation_runtime,
        )

        if any(
            not value.strip()
            for value in required_values
        ):
            raise ValueError(
                "Generated-answer fields cannot be empty."
            )

        if (
            not self.refused
            and not self.insufficient_context
            and not self.citation_validation.valid
        ):
            raise ValueError(
                "A substantive answer must contain only "
                "known citations."
            )

        return self


def extract_answer_citations(
    answer_text: str,
) -> tuple[str, ...]:
    """Extract unique citations in first-occurrence order."""

    return tuple(
        dict.fromkeys(
            _CITATION_PATTERN.findall(
                answer_text
            )
        )
    )


def validate_answer_citations(
    *,
    answer_text: str,
    prompt: GenerationPrompt,
) -> AnswerCitationValidation:
    """Validate answer citations against prompt sources."""

    occurrences = _CITATION_PATTERN.findall(
        answer_text
    )

    used_citation_ids = tuple(
        dict.fromkeys(occurrences)
    )

    available_citation_ids = (
        prompt.citation_ids
    )

    available = set(
        available_citation_ids
    )
    used = set(
        used_citation_ids
    )
    unknown_citation_ids = tuple(
        citation_id
        for citation_id in used_citation_ids
        if citation_id not in available
    )

    unused_citation_ids = tuple(
        citation_id
        for citation_id
        in available_citation_ids
        if citation_id not in used
    )

    has_citations = bool(
        used_citation_ids
    )

    all_citations_known = not (
        unknown_citation_ids
    )

    return AnswerCitationValidation(
        available_citation_ids=(
            available_citation_ids
        ),
        used_citation_ids=(
            used_citation_ids
        ),
        unknown_citation_ids=(
            unknown_citation_ids
        ),
        unused_citation_ids=(
            unused_citation_ids
        ),
        citation_occurrence_count=len(
            occurrences
        ),
        has_citations=has_citations,
        all_citations_known=(
            all_citations_known
        ),
        valid=(
            has_citations
            and all_citations_known
        ),
    )


def build_generated_answer(
    *,
    request: GenerationRequest,
    response: GenerationRuntimeResponse,
    refused: bool = False,
    insufficient_context: bool = False,
    metadata: dict[str, object] | None = None,
) -> GeneratedAnswer:
    """Build a validated answer from a normalized response."""

    prompt = request.prompt

    citation_validation = (
        validate_answer_citations(
            answer_text=response.answer_text,
            prompt=prompt,
        )
    )

    identity = {
        "prompt_id": prompt.prompt_id,
        "model_id": response.model_id,
        "model_revision": (
            response.model_revision
        ),
        "runtime_id": response.runtime_id,
        "answer_text": response.answer_text,
        "generation_parameters": {
            "maximum_output_tokens": (
                request.maximum_output_tokens
            ),
            "temperature": request.temperature,
            "seed": request.seed,
            "response_format": (
                request.response_format
            ),
        },
    }

    fingerprint = sha256_fingerprint(identity)[:12]
    answer_id = (
        f"answer-{prompt.dataset_id}-"
        f"{prompt.query_id}-{fingerprint}"
    )

    return GeneratedAnswer(
        answer_id=answer_id,
        prompt_id=prompt.prompt_id,
        context_pack_id=prompt.context_pack_id,
        query_id=prompt.query_id,
        dataset_id=prompt.dataset_id,
        answer_text=response.answer_text,
        generation_model_id=response.model_id,
        generation_model_revision=(
            response.model_revision
        ),
        generation_runtime=(
            response.runtime_id
        ),
        input_token_count=(
            response.input_token_count
        ),
        output_token_count=(
            response.output_token_count
        ),
        finish_reason=(
            response.finish_reason
        ),
        refused=refused,
        insufficient_context=(
            insufficient_context
        ),
        citation_validation=(
            citation_validation
        ),
        generation_parameters={
            "maximum_output_tokens": (
                request.maximum_output_tokens
            ),
            "temperature": request.temperature,
            "seed": request.seed,
            "response_format": (
                request.response_format
            ),
        },
        metadata={
            **(
                metadata
                if metadata is not None
                else {}
            ),
            "runtime_request_id": (
                response.request_id
            ),
            "runtime_metadata": (
                response.metadata
                if response.metadata is not None
                else {}
            ),
        },
    )
