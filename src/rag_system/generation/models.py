"""Typed, citation-preserving generation context artifacts."""

from __future__ import annotations

from pydantic import Field, model_validator

from rag_system.artifacts import ArtifactBaseModel


class ContextItem(ArtifactBaseModel):
    """One retrieved source included in a generation context."""

    citation_id: str

    chunk_id: str
    parent_id: str
    doc_id: str
    dataset: str

    retrieval_rank: int = Field(gt=0)
    retrieval_score: float
    retrieval_method: str

    text: str
    token_count: int = Field(gt=0)

    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_context_item(self) -> ContextItem:
        """Require complete source and citation identity."""

        required_values = (
            self.citation_id,
            self.chunk_id,
            self.parent_id,
            self.doc_id,
            self.dataset,
            self.retrieval_method,
            self.text,
        )

        if any(not value.strip() for value in required_values):
            raise ValueError("Context-item identity fields cannot be empty.")

        if not (self.citation_id.startswith("[C") and self.citation_id.endswith("]")):
            raise ValueError("citation_id must use the form [C<number>].")

        return self


class ContextPack(ArtifactBaseModel):
    """Deterministic source context for one generation query."""

    artifact_type: str = "generation_context_pack"
    artifact_version: str = "1.0"

    context_pack_id: str

    query_id: str
    query_text: str
    dataset_id: str

    retrieval_method: str

    tokenizer_id: str
    maximum_context_tokens: int = Field(gt=0)
    used_context_tokens: int = Field(ge=0)
    remaining_context_tokens: int = Field(ge=0)

    selected_item_count: int = Field(ge=0)
    skipped_item_count: int = Field(ge=0)

    selected_items: tuple[ContextItem, ...]
    skipped_chunk_ids: tuple[str, ...]

    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_context_pack(self) -> ContextPack:
        """Validate token accounting and source uniqueness."""

        required_values = (
            self.context_pack_id,
            self.query_id,
            self.query_text,
            self.dataset_id,
            self.retrieval_method,
            self.tokenizer_id,
        )

        if any(not value.strip() for value in required_values):
            raise ValueError("Context-pack identity fields cannot be empty.")

        if (
            self.used_context_tokens + self.remaining_context_tokens
            != self.maximum_context_tokens
        ):
            raise ValueError("Context-pack token accounting is inconsistent.")

        if len(self.selected_items) != self.selected_item_count:
            raise ValueError("selected_item_count does not match selected_items.")

        if len(self.skipped_chunk_ids) != self.skipped_item_count:
            raise ValueError("skipped_item_count does not match skipped IDs.")

        selected_chunk_ids = [item.chunk_id for item in self.selected_items]

        if len(selected_chunk_ids) != len(set(selected_chunk_ids)):
            raise ValueError("Context pack contains duplicate chunk IDs.")

        citation_ids = [item.citation_id for item in self.selected_items]

        expected_citations = [
            f"[C{index}]" for index in range(1, len(self.selected_items) + 1)
        ]

        if citation_ids != expected_citations:
            raise ValueError("Context citations must be contiguous and ordered.")

        calculated_token_count = sum(
            item.token_count for item in self.selected_items
        )

        if calculated_token_count != self.used_context_tokens:
            raise ValueError("Context item tokens do not match used_context_tokens.")

        selected_set = set(selected_chunk_ids)

        if selected_set.intersection(self.skipped_chunk_ids):
            raise ValueError("A chunk cannot be both selected and skipped.")

        return self
