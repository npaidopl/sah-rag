"""Typed models for canonical benchmark records."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

JsonObject = dict[str, Any]


class GroundTruthType(StrEnum):
    """Known canonical ground-truth families."""

    COMPOSITE = "composite"
    SPAN_SET = "span_set"
    FREE_TEXT = "free_text"
    MULTIPLE_CHOICE = "multiple_choice"
    DETERMINISTIC = "deterministic"


class CanonicalBaseModel(BaseModel):
    """Common configuration for canonical models."""

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        str_strip_whitespace=False,
    )


class CanonicalDocument(CanonicalBaseModel):
    """One immutable canonical document record."""

    doc_id: str
    dataset: str
    text_raw: str
    text_normalized: str
    title: str | None = None
    source_path: str | None = None
    source_row: int | str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("doc_id", "dataset")
    @classmethod
    def validate_non_empty_identifier(
        cls,
        value: str,
    ) -> str:
        """Require non-empty canonical identifiers."""

        if not value.strip():
            raise ValueError("Canonical identifier cannot be empty.")

        return value


class CanonicalQuery(CanonicalBaseModel):
    """Canonical query embedded inside an example record."""

    query_id: str
    dataset: str
    question: str
    task_type: str | None = None
    split: str | None = None
    doc_ids: list[str] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("query_id", "dataset")
    @classmethod
    def validate_non_empty_identifier(
        cls,
        value: str,
    ) -> str:
        """Require non-empty query identifiers."""

        if not value.strip():
            raise ValueError("Canonical identifier cannot be empty.")

        return value

    @field_validator("doc_ids")
    @classmethod
    def validate_document_references(
        cls,
        values: list[str],
    ) -> list[str]:
        """Reject empty document references."""

        if any(not value.strip() for value in values):
            raise ValueError(
                "query.doc_ids cannot contain empty identifiers."
            )

        return values


class CanonicalGroundTruth(CanonicalBaseModel):
    """Canonical ground truth with dataset-specific optional fields."""

    ground_truth_type: GroundTruthType
    dataset: str
    query_id: str
    answers: list[Any] = Field(default_factory=list)
    relevant_ids: list[str] = Field(default_factory=list)
    choice: Any | None = None
    spans: list[Any] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("dataset", "query_id")
    @classmethod
    def validate_non_empty_identifier(
        cls,
        value: str,
    ) -> str:
        """Require non-empty ground-truth identifiers."""

        if not value.strip():
            raise ValueError("Canonical identifier cannot be empty.")

        return value

    @field_validator("relevant_ids")
    @classmethod
    def validate_relevant_ids(
        cls,
        values: list[str],
    ) -> list[str]:
        """Reject empty relevant-document identifiers."""

        if any(not value.strip() for value in values):
            raise ValueError(
                "ground_truth.relevant_ids cannot contain empty IDs."
            )

        return values


class CanonicalExample(CanonicalBaseModel):
    """One canonical benchmark example."""

    query: CanonicalQuery
    ground_truth: CanonicalGroundTruth
    supplied_context: Any | None = None
    options: Any | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("ground_truth")
    @classmethod
    def validate_query_ground_truth_identity(
        cls,
        ground_truth: CanonicalGroundTruth,
        info: Any,
    ) -> CanonicalGroundTruth:
        """Identity consistency is finalized at model level."""

        return ground_truth

    def model_post_init(
        self,
        __context: Any,
    ) -> None:
        """Verify query and ground-truth identity consistency."""

        if self.query.query_id != self.ground_truth.query_id:
            raise ValueError(
                "query.query_id does not match "
                "ground_truth.query_id: "
                f"{self.query.query_id!r} != "
                f"{self.ground_truth.query_id!r}"
            )

        if self.query.dataset != self.ground_truth.dataset:
            raise ValueError(
                "query.dataset does not match "
                "ground_truth.dataset: "
                f"{self.query.dataset!r} != "
                f"{self.ground_truth.dataset!r}"
            )