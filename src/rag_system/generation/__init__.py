"""Context assembly and answer-generation artifacts."""

from rag_system.generation.adapters import (
    DeterministicDryRunAdapter,
)
from rag_system.generation.answers import (
    AnswerCitationValidation,
    GeneratedAnswer,
    build_generated_answer,
    extract_answer_citations,
    validate_answer_citations,
)
from rag_system.generation.context import (
    TokenCounter,
    build_context_pack,
)
from rag_system.generation.identity import (
    build_context_pack_id,
)
from rag_system.generation.models import (
    ContextItem,
    ContextPack,
)
from rag_system.generation.prompts import (
    GenerationPrompt,
    PromptTokenCounter,
    build_generation_prompt,
    render_context,
)
from rag_system.generation.runtime import (
    GenerationRequest,
    GenerationRuntimeAdapter,
    GenerationRuntimeResponse,
)

__all__ = [
    "AnswerCitationValidation",
    "ContextItem",
    "ContextPack",
    "DeterministicDryRunAdapter",
    "GeneratedAnswer",
    "GenerationPrompt",
    "GenerationRequest",
    "GenerationRuntimeAdapter",
    "GenerationRuntimeResponse",
    "PromptTokenCounter",
    "TokenCounter",
    "build_context_pack",
    "build_context_pack_id",
    "build_generated_answer",
    "build_generation_prompt",
    "extract_answer_citations",
    "render_context",
    "validate_answer_citations",
]
