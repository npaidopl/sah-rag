"""Typed retrieval results and parent aggregations."""

from __future__ import annotations

from pydantic import Field, model_validator

from rag_system.artifacts.models import ArtifactBaseModel


class RetrievedChunk(ArtifactBaseModel):
    """One retrieved child chunk."""

    chunk_id: str
    parent_id: str
    doc_id: str
    dataset: str

    rank: int = Field(ge=1)
    score: float

    text_normalized: str
    retrieval_method: str

    payload: dict[str, object] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_identity(self) -> RetrievedChunk:
        """Require complete retrieval identity."""
        required = (
            self.chunk_id,
            self.parent_id,
            self.doc_id,
            self.dataset,
            self.retrieval_method,
        )

        if any(not value.strip() for value in required):
            raise ValueError(
                "Retrieved-chunk identity fields cannot be empty."
            )

        return self


class RetrievedParent(ArtifactBaseModel):
    """Parent-level aggregation of retrieved child chunks."""

    parent_id: str
    doc_id: str
    dataset: str

    rank: int = Field(ge=1)
    score: float

    best_chunk_rank: int = Field(ge=1)
    matched_chunk_count: int = Field(ge=1)

    chunk_ids: tuple[str, ...]
    child_scores: tuple[float, ...]

    @model_validator(mode="after")
    def validate_aggregation(self) -> RetrievedParent:
        """Validate aligned child evidence."""
        if len(self.chunk_ids) != len(self.child_scores):
            raise ValueError(
                "Parent chunk IDs and scores must align."
            )

        if len(self.chunk_ids) != self.matched_chunk_count:
            raise ValueError(
                "matched_chunk_count does not match evidence."
            )

        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError(
                "Parent aggregation contains duplicate chunks."
            )

        return self
