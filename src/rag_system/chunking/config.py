"""Configuration models for hierarchical child chunking."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChildChunkerConfig(BaseModel):
    """Configuration of the parent-aware token chunker."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunker_id: str = "hierarchical_token_chunker"
    chunker_version: str = "0.3.0"

    chunk_size: int = Field(
        default=256,
        gt=0,
        description=(
            "Total model-input token budget, including "
            "sequence-level special tokens."
        ),
    )

    chunk_overlap: int = Field(
        default=32,
        ge=0,
        description=(
            "Overlap between content-token windows."
        ),
    )

    minimum_final_chunk_tokens: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Minimum content tokens in an anchored final window."
        ),
    )

    preserve_parent_boundaries: bool = True
    include_empty_parents: bool = False

    @model_validator(mode="after")
    def validate_configuration(
        self,
    ) -> ChildChunkerConfig:
        """Validate model-independent configuration."""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if (
            self.minimum_final_chunk_tokens is not None
            and self.minimum_final_chunk_tokens
            > self.chunk_size
        ):
            raise ValueError(
                "minimum_final_chunk_tokens cannot exceed "
                "chunk_size."
            )

        if not self.preserve_parent_boundaries:
            raise ValueError(
                "The hierarchical chunker must preserve "
                "semantic-parent boundaries."
            )

        return self

    def content_chunk_size(
        self,
        *,
        special_tokens_count: int,
    ) -> int:
        """Return usable content tokens per model input."""

        if special_tokens_count < 0:
            raise ValueError(
                "special_tokens_count cannot be negative."
            )

        content_size = (
            self.chunk_size - special_tokens_count
        )

        if content_size < 1:
            raise ValueError(
                "chunk_size must exceed the tokenizer's "
                "special-token count."
            )

        return content_size

    def effective_minimum_final_chunk_tokens(
        self,
        *,
        content_chunk_size: int,
    ) -> int:
        """Return minimum content tokens in the final window."""

        if content_chunk_size < 1:
            raise ValueError(
                "content_chunk_size must be positive."
            )

        if self.minimum_final_chunk_tokens is not None:
            if (
                self.minimum_final_chunk_tokens
                > content_chunk_size
            ):
                raise ValueError(
                    "minimum_final_chunk_tokens cannot exceed "
                    "the available content-token capacity."
                )

            return self.minimum_final_chunk_tokens

        return max(1, content_chunk_size // 2)
