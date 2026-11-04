"""Context assembly and answer-generation artifacts."""

from sah_rag.generation.adapters import (
    DeterministicDryRunAdapter,
)
from sah_rag.generation.answers import (
    AnswerCitationValidation,
    GeneratedAnswer,
    build_generated_answer,
    extract_answer_citations,
    validate_answer_citations,
)
from sah_rag.generation.context import (
    TokenCounter,
    build_context_pack,
)
from sah_rag.generation.identity import (
    build_context_pack_id,
)
from sah_rag.generation.models import (
    ContextItem,
    ContextPack,
)
from sah_rag.generation.pipeline import (
    PipelineConfig,
    answer,
)
from sah_rag.generation.prompts import (
    GenerationPrompt,
    PromptTokenCounter,
    build_generation_prompt,
    render_context,
)
from sah_rag.generation.qwen_adapter import (
    QwenLLMAdapter,
)
from sah_rag.generation.runtime import (
    GenerationRequest,
    GenerationRuntimeAdapter,
    GenerationRuntimeResponse,
)

__all__ = [
    # Data models
    "AnswerCitationValidation",
    "ContextItem",
    "ContextPack",
    "GeneratedAnswer",
    "GenerationPrompt",
    # Runtime contracts
    "GenerationRequest",
    "GenerationRuntimeAdapter",
    "GenerationRuntimeResponse",
    # Runtimes / adapters
    "DeterministicDryRunAdapter",
    "QwenLLMAdapter",
    # Token counters
    "PromptTokenCounter",
    "TokenCounter",
    # Pipeline
    "PipelineConfig",
    "answer",
    # Builders
    "build_context_pack",
    "build_context_pack_id",
    "build_generated_answer",
    "build_generation_prompt",
    # Utilities
    "extract_answer_citations",
    "render_context",
    "validate_answer_citations",
]