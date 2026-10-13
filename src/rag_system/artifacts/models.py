"""Typed models for derived RAG artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

JsonObject = dict[str, Any]


class ArtifactBaseModel(BaseModel):
    """Common configuration for derived artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )


class BoundaryMethod(StrEnum):
    """Supported semantic-boundary provenance values."""

    DOCUMENT_PREAMBLE = "dpcument_preamble"
    DOCUMENT_START = "document_start"
    DOCUMENT_END = "document_end"
    EXPLICIT_HEADING = "explicit_heading"
    NUMBERED_HIERARCHY = "numbered_hierarchy"
    PARAGRAPH_BREAK = "paragraph_break"
    STRUCTURAL_RULE = "structural_rule"
    SEMANTIC_DROP = "semantic_drop"
    STATISTICAL_CLASSIFIER = "statistical_classifier"
    MAXIMUM_LENGTH_FALLBACK = "maximum_length_fallback"
    PARAGRAPH_GROUP_FALLBACK = "paragraph_group_fallback"


class OffsetSpan(ArtifactBaseModel):
    """Half-open character or token span."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> OffsetSpan:
        """Require a non-negative, non-reversed span."""

        if self.end < self.start:
            raise ValueError(
                "Offset span end cannot be smaller than start."
            )

        return self

    @property
    def length(self) -> int:
        """Return the span length."""

        return self.end - self.start


class BoundaryEvidence(ArtifactBaseModel):
    """Evidence supporting one predicted semantic boundary."""

    method: BoundaryMethod
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    score: float | None = None
    rule_id: str | None = None
    details: JsonObject = Field(default_factory=dict)


class SemanticParent(ArtifactBaseModel):
    """One parsed semantic entity derived from a canonical document."""

    parent_id: str
    doc_id: str
    dataset: str

    parent_parent_id: str | None = None
    hierarchy_level: int = Field(ge=0)
    sibling_index: int = Field(ge=0)

    title: str | None = None
    text_normalized: str
    text_raw: str | None = None

    normalized_char_span: OffsetSpan
    raw_char_span: OffsetSpan | None = None

    token_span: OffsetSpan | None = None
    token_count: int | None = Field(default=None, ge=0)

    start_boundary: tuple[BoundaryEvidence, ...] = ()
    end_boundary: tuple[BoundaryEvidence, ...] = ()

    parser_id: str
    parser_version: str
    parser_config_hash: str

    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parent(self) -> SemanticParent:
        """Validate semantic-parent identity and span consistency."""

        required_identifiers = (
            self.parent_id,
            self.doc_id,
            self.dataset,
            self.parser_id,
            self.parser_version,
            self.parser_config_hash,
        )

        if any(not value.strip() for value in required_identifiers):
            raise ValueError(
                "Semantic-parent identifiers cannot be empty."
            )

        if (
            self.parent_parent_id is not None
            and self.parent_parent_id == self.parent_id
        ):
            raise ValueError(
                "A semantic parent cannot be its own parent."
            )

        if not self.text_normalized and (
            self.normalized_char_span.length > 0
        ):
            raise ValueError(
                "A non-empty normalized span requires text."
            )

        return self


class ParsedDocument(ArtifactBaseModel):
    """Complete parsed representation of one canonical document."""

    artifact_type: str = "parsed_document"
    artifact_version: str = "1.0"

    doc_id: str
    dataset: str

    parser_id: str
    parser_version: str
    parser_config_hash: str

    canonical_text_normalized_length: int = Field(ge=0)
    canonical_text_raw_length: int = Field(ge=0)

    parents: tuple[SemanticParent, ...]

    source_fingerprint: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parsed_document(self) -> ParsedDocument:
        """Validate parent identity, uniqueness, and document links."""

        if not self.doc_id.strip() or not self.dataset.strip():
            raise ValueError(
                "Parsed-document identifiers cannot be empty."
            )

        parent_ids = [
            parent.parent_id
            for parent in self.parents
        ]

        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError(
                "Parsed document contains duplicate parent IDs."
            )

        for parent in self.parents:
            if (
                parent.normalized_char_span.end
                > self.canonical_text_normalized_length
            ):
                raise ValueError(
                    "Semantic-parent normalized span exceeds "
                    "the canonical normalized-text length."
                )

            if (
                parent.raw_char_span is not None
                and parent.raw_char_span.end
                > self.canonical_text_raw_length
            ):
                raise ValueError(
                    "Semantic-parent raw span exceeds "
                    "the canonical raw-text length."
                )

            if parent.doc_id != self.doc_id:
                raise ValueError(
                    "Semantic parent references a different doc_id."
                )

            if parent.dataset != self.dataset:
                raise ValueError(
                    "Semantic parent references a different dataset."
                )

            if parent.parser_id != self.parser_id:
                raise ValueError(
                    "Semantic parent parser_id does not match "
                    "the parsed document."
                )

            if parent.parser_version != self.parser_version:
                raise ValueError(
                    "Semantic parent parser_version does not match "
                    "the parsed document."
                )

            if (
                parent.parser_config_hash
                != self.parser_config_hash
            ):
                raise ValueError(
                    "Semantic parent parser configuration does not "
                    "match the parsed document."
                )

        known_parent_ids = set(parent_ids)

        for parent in self.parents:
            ancestor_id = parent.parent_parent_id

            if (
                ancestor_id is not None
                and ancestor_id not in known_parent_ids
            ):
                raise ValueError(
                    "Semantic parent references an unknown "
                    "parent_parent_id."
                )

        return self


class ChildChunk(ArtifactBaseModel):
    """One indexed retrieval chunk derived from a semantic parent."""

    artifact_type: str = "child_chunk"
    artifact_version: str = "1.0"

    chunk_id: str
    parent_id: str
    doc_id: str
    dataset: str

    chunk_index: int = Field(ge=0)
    parent_chunk_index: int = Field(ge=0)

    text_normalized: str
    text_raw: str | None = None

    normalized_char_span: OffsetSpan
    parent_normalized_char_span: OffsetSpan

    raw_char_span: OffsetSpan | None = None
    token_span: OffsetSpan | None = None
    token_count: int = Field(ge=0)

    parser_id: str
    parser_version: str
    parser_config_hash: str

    chunker_id: str
    chunker_version: str
    chunker_config_hash: str

    tokenizer_id: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_chunk(self) -> ChildChunk:
        """Validate child identity, spans, and chunk configuration."""

        required_identifiers = (
            self.chunk_id,
            self.parent_id,
            self.doc_id,
            self.dataset,
            self.parser_id,
            self.parser_version,
            self.parser_config_hash,
            self.chunker_id,
            self.chunker_version,
            self.chunker_config_hash,
            self.tokenizer_id,
        )

        if any(not value.strip() for value in required_identifiers):
            raise ValueError(
                "Child-chunk identifiers cannot be empty."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if (
            self.normalized_char_span.start
            < self.parent_normalized_char_span.start
            or self.normalized_char_span.end
            > self.parent_normalized_char_span.end
        ):
            raise ValueError(
                "Child normalized span must be contained inside "
                "the semantic-parent span."
            )

        if not self.text_normalized and self.token_count > 0:
            raise ValueError(
                "A chunk with tokens must contain normalized text."
            )

        if (
            self.token_span is not None
            and self.token_span.length != self.token_count
        ):
            raise ValueError(
                "token_count must equal the token-span length."
            )

        return self