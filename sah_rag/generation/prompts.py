"""Citation-preserving prompt artifacts and rendering."""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import Field, model_validator

from rag_system.artifacts import (
    ArtifactBaseModel,
    sha256_fingerprint,
)
from rag_system.generation.models import (
    ContextPack,
)

PromptTokenCounter = Callable[[str], int]

_CITATION_PATTERN = re.compile(r"\[C[1-9][0-9]*\]")


class GenerationPrompt(ArtifactBaseModel):
    """A deterministic prompt rendered from one context pack."""

    artifact_type: str = "generation_prompt"
    artifact_version: str = "1.0"

    prompt_id: str
    context_pack_id: str

    query_id: str
    dataset_id: str

    template_id: str
    template_version: str

    tokenizer_id: str

    source_context_tokens: int = Field(ge=0)
    rendered_prompt_tokens: int = Field(gt=0)
    maximum_prompt_tokens: int = Field(gt=0)
    reserved_answer_tokens: int = Field(ge=0)

    system_instruction: str
    rendered_context: str
    user_prompt: str

    citation_ids: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]
    source_document_ids: tuple[str, ...]

    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prompt(self) -> GenerationPrompt:
        """Validate prompt budget and citation integrity."""

        required_values = (
            self.prompt_id,
            self.context_pack_id,
            self.query_id,
            self.dataset_id,
            self.template_id,
            self.template_version,
            self.tokenizer_id,
            self.system_instruction,
            self.rendered_context,
            self.user_prompt,
        )

        if any(not value.strip() for value in required_values):
            raise ValueError("Generation-prompt fields cannot be empty.")

        if (
            self.rendered_prompt_tokens + self.reserved_answer_tokens
            > self.maximum_prompt_tokens
        ):
            raise ValueError(
                "Prompt and reserved answer tokens exceed "
                "the maximum prompt budget."
            )

        if len(self.citation_ids) != len(self.source_chunk_ids):
            raise ValueError("Citation and source-chunk counts must align.")

        if len(self.citation_ids) != len(set(self.citation_ids)):
            raise ValueError("Prompt citation IDs must be unique.")

        rendered_citations = tuple(
            dict.fromkeys(_CITATION_PATTERN.findall(self.rendered_context))
        )

        if rendered_citations != self.citation_ids:
            raise ValueError(
                "Rendered citations do not match the declared citation IDs."
            )

        for citation_id in self.citation_ids:
            if citation_id not in self.user_prompt:
                raise ValueError(
                    f"Missing citation in prompt: {citation_id}"
                )

        return self


def render_context(context_pack: ContextPack) -> str:
    """Render context items with stable source metadata."""

    sections: list[str] = []

    for item in context_pack.selected_items:
        sections.append(
            "\n".join(
                (
                    item.citation_id,
                    f"document_id: {item.doc_id}",
                    f"chunk_id: {item.chunk_id}",
                    "source_text:",
                    item.text,
                )
            )
        )

    return "\n\n".join(sections)


def build_generation_prompt(
    *,
    context_pack: ContextPack,
    token_counter: PromptTokenCounter,
    maximum_prompt_tokens: int,
    reserved_answer_tokens: int,
    system_instruction: str | None = None,
    template_id: str = "clerk_cited_qa",
    template_version: str = "1.0.0",
    metadata: dict[str, object] | None = None,
) -> GenerationPrompt:
    """Render a deterministic cited-answer prompt."""

    if maximum_prompt_tokens < 1:
        raise ValueError("maximum_prompt_tokens must be positive.")

    if reserved_answer_tokens < 0:
        raise ValueError("reserved_answer_tokens cannot be negative.")

    if reserved_answer_tokens >= maximum_prompt_tokens:
        raise ValueError(
            "reserved_answer_tokens must be smaller than maximum_prompt_tokens."
        )

    instruction = system_instruction or (
        "Answer the question using only the supplied sources. "
        "Every factual claim must be supported by one or more "
        "citations in the form [C1], [C2], and so on. "
        "Do not invent citations. "
        "If the sources do not support an answer, state that "
        "the available sources are insufficient."
    )

    rendered_ctx = render_context(context_pack)

    user_prompt = "\n\n".join(
        (
            "Question:",
            context_pack.query_text,
            "Sources:",
            rendered_ctx,
            "Answer requirements:",
            (
                "Provide a concise answer grounded only in "
                "the sources above. Include source citations "
                "after the claims they support."
            ),
        )
    )

    full_prompt_for_counting = "\n\n".join(
        (
            instruction,
            user_prompt,
        )
    )

    rendered_prompt_tokens = token_counter(full_prompt_for_counting)

    if rendered_prompt_tokens < 1:
        raise ValueError("Prompt token counter returned a non-positive value.")

    if rendered_prompt_tokens + reserved_answer_tokens > maximum_prompt_tokens:
        raise ValueError(
            "The rendered prompt exceeds the available "
            "prompt budget after reserving answer tokens."
        )

    citation_ids = tuple(item.citation_id for item in context_pack.selected_items)
    source_chunk_ids = tuple(item.chunk_id for item in context_pack.selected_items)
    source_document_ids = tuple(
        dict.fromkeys(item.doc_id for item in context_pack.selected_items)
    )

    identity = {
        "context_pack_id": context_pack.context_pack_id,
        "template_id": template_id,
        "template_version": template_version,
        "system_instruction": instruction,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "reserved_answer_tokens": reserved_answer_tokens,
        "rendered_prompt_sha256": sha256_fingerprint(full_prompt_for_counting),
    }

    prompt_id = (
        "prompt-"
        f"{context_pack.dataset_id}-"
        f"{context_pack.query_id}-"
        f"{sha256_fingerprint(identity)[:12]}"
    )

    return GenerationPrompt(
        prompt_id=prompt_id,
        context_pack_id=context_pack.context_pack_id,
        query_id=context_pack.query_id,
        dataset_id=context_pack.dataset_id,
        template_id=template_id,
        template_version=template_version,
        tokenizer_id=context_pack.tokenizer_id,
        source_context_tokens=context_pack.used_context_tokens,
        rendered_prompt_tokens=rendered_prompt_tokens,
        maximum_prompt_tokens=maximum_prompt_tokens,
        reserved_answer_tokens=reserved_answer_tokens,
        system_instruction=instruction,
        rendered_context=rendered_ctx,
        user_prompt=user_prompt,
        citation_ids=citation_ids,
        source_chunk_ids=source_chunk_ids,
        source_document_ids=source_document_ids,
        metadata=metadata if metadata is not None else {},
    )
