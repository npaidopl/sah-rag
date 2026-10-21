"""Deterministic metadata extraction for the Metadata-RAG comparator.

Frozen specification:
    baseline_audit/metadata_rag_clean_room_spec.yaml
Frozen amendment 01:
    baseline_audit/metadata_rag_clean_room_spec_amendment_01.yaml

The extractor is symmetric: the same routines are applied to document
text and to query text. This symmetry is the architecture-defining
property that the frozen clean-room spec relies on, and it is the
reason the query side may never read doc_ids, relevant_ids, answers,
target_content_group_id, ground_truth, spans, or query metadata.

Unicode scope: the frozen patterns in this module operate on the ASCII
ranges ``[A-Z][a-z]``. Datasets whose text uses non-ASCII capitals
(for example Greek Bar Bench) will produce empty
``capitalized_phrases`` records under schema_version 0.1.0. That gap
is documented and not silently patched.
"""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.1.0"
EXTRACTOR_VERSION = "0.1.0"

CAPITALIZED_PHRASE_PATTERN = (
    r"(?<!\S)(?!(?:The|A|An)\b)(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
)
YEAR_MENTION_PATTERN = r"\b(?:19|20)\d{2}\b"
NUMERIC_TOKEN_PATTERN = r"\b\d+(?:\.\d+)?\b"

SHORT_MAX_WORDS = 80
MEDIUM_MAX_WORDS = 400

_CAPITALIZED_PHRASE_RE = re.compile(CAPITALIZED_PHRASE_PATTERN)
_YEAR_RE = re.compile(YEAR_MENTION_PATTERN)
_NUMERIC_RE = re.compile(NUMERIC_TOKEN_PATTERN)

LengthBucket = Literal["short", "medium", "long"]
SourceKind = Literal["document", "query"]
ConstraintProvenance = Literal["QUERY_TEXT"]


def extract_capitalized_phrases(text: str) -> tuple[str, ...]:
    """Return lexically-sorted, deduplicated, lowercased multi-word names."""
    matches = {m.strip().lower() for m in _CAPITALIZED_PHRASE_RE.findall(text)}
    return tuple(sorted(matches))


def extract_year_mentions(text: str) -> tuple[int, ...]:
    """Return numerically-sorted, deduplicated year mentions."""
    matches = {int(m) for m in _YEAR_RE.findall(text)}
    return tuple(sorted(matches))


def extract_numeric_tokens(text: str) -> tuple[str, ...]:
    """Return lexically-sorted, deduplicated numeric tokens in string form."""
    matches = set(_NUMERIC_RE.findall(text))
    return tuple(sorted(matches))


def compute_word_count(text: str) -> int:
    """Whitespace token count."""
    return len(text.split())


def compute_length_bucket(word_count: int) -> LengthBucket:
    """Bucket the word count per the frozen thresholds."""
    if word_count <= SHORT_MAX_WORDS:
        return "short"
    if word_count <= MEDIUM_MAX_WORDS:
        return "medium"
    return "long"


class MetadataRecord(BaseModel):
    """One frozen Metadata-RAG metadata record.

    The record shape is invariant across documents and queries. The
    ``source_kind`` discriminator decides which tier is required.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    source_kind: SourceKind
    dataset: str

    doc_id: str | None = None
    source_dataset_id: str | None = None
    source_content_group_id: str | None = None
    title: str | None = None

    capitalized_phrases: tuple[str, ...] = ()
    year_mentions: tuple[int, ...] = ()
    numeric_tokens: tuple[str, ...] = ()
    char_length: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    length_bucket: LengthBucket = "short"

    schema_version: str = SCHEMA_VERSION
    extractor_version: str = EXTRACTOR_VERSION

    @model_validator(mode="after")
    def _enforce_provenance_tier(self) -> Self:
        if self.source_kind == "document":
            if not self.doc_id:
                raise ValueError("document records require doc_id")
            if not self.source_dataset_id:
                raise ValueError("document records require source_dataset_id")
        return self


def _normalize_dataset(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("dataset cannot be empty")
    return normalized


def _normalize_as_is(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _normalize_optional_title(title: str | None) -> str | None:
    if title is None:
        return None
    normalized = title.strip().lower()
    return normalized or None


def build_document_metadata(
    *,
    dataset: str,
    doc_id: str,
    source_dataset_id: str,
    text_normalized: str,
    source_content_group_id: str | None = None,
    title: str | None = None,
) -> MetadataRecord:
    """Build a frozen metadata record for one canonical document."""
    if not text_normalized.strip():
        raise ValueError("text_normalized cannot be empty")

    word_count = compute_word_count(text_normalized)
    return MetadataRecord(
        source_kind="document",
        dataset=_normalize_dataset(dataset),
        doc_id=_normalize_as_is(doc_id, field_name="doc_id"),
        source_dataset_id=_normalize_as_is(
            source_dataset_id, field_name="source_dataset_id"
        ),
        source_content_group_id=source_content_group_id,
        title=_normalize_optional_title(title),
        capitalized_phrases=extract_capitalized_phrases(text_normalized),
        year_mentions=extract_year_mentions(text_normalized),
        numeric_tokens=extract_numeric_tokens(text_normalized),
        char_length=len(text_normalized),
        word_count=word_count,
        length_bucket=compute_length_bucket(word_count),
    )


def build_query_metadata(
    *,
    dataset: str,
    query_id: str,
    query_text: str,
) -> MetadataRecord:
    """Build a frozen metadata record from query text only.

    Reading any other query field -- ``doc_ids``, ``relevant_ids``,
    ``answers``, ``target_content_group_id``, ``ground_truth``,
    ``spans``, or ``metadata`` -- would leak ground truth into
    retrieval and is prohibited by the frozen spec.
    """
    if not query_id.strip():
        raise ValueError("query_id cannot be empty")
    if not query_text.strip():
        raise ValueError("query_text cannot be empty")

    word_count = compute_word_count(query_text)
    return MetadataRecord(
        source_kind="query",
        dataset=_normalize_dataset(dataset),
        capitalized_phrases=extract_capitalized_phrases(query_text),
        year_mentions=extract_year_mentions(query_text),
        numeric_tokens=extract_numeric_tokens(query_text),
        char_length=len(query_text),
        word_count=word_count,
        length_bucket=compute_length_bucket(word_count),
    )


class MetadataConstraint(BaseModel):
    """One query-derived metadata constraint."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    constraint_id: str
    field: str
    value: str
    provenance: ConstraintProvenance = "QUERY_TEXT"


CONSTRAINT_FIELDS: tuple[str, ...] = (
    "capitalized_phrases",
    "year_mentions",
    "numeric_tokens",
)


def derive_query_constraints(
    record: MetadataRecord,
) -> tuple[MetadataConstraint, ...]:
    """Convert a query MetadataRecord into typed constraints.

    Only the multi-value derived fields produce constraints. The scalar
    fields (``char_length``, ``word_count``, ``length_bucket``) are not
    entity-level signals and are therefore not turned into constraints.
    """
    if record.source_kind != "query":
        raise ValueError("derive_query_constraints requires a query record")

    constraints: list[MetadataConstraint] = []
    for phrase in record.capitalized_phrases:
        constraints.append(
            MetadataConstraint(
                constraint_id=f"cap:{phrase}",
                field="capitalized_phrases",
                value=phrase,
            )
        )
    for year in record.year_mentions:
        constraints.append(
            MetadataConstraint(
                constraint_id=f"year:{year}",
                field="year_mentions",
                value=str(year),
            )
        )
    for token in record.numeric_tokens:
        constraints.append(
            MetadataConstraint(
                constraint_id=f"num:{token}",
                field="numeric_tokens",
                value=token,
            )
        )
    return tuple(constraints)
