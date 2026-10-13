"""Configuration models for deterministic parsing."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DeterministicParserConfig(BaseModel):
    """Configuration of the deterministic structural parser."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    parser_id: str = "deterministic_structural_parser"
    parser_version: str = "0.3.0"

    detect_numbered_headings: bool = True
    detect_named_headings: bool = True
    detect_uppercase_headings: bool = True

    maximum_heading_characters: int = Field(
        default=120,
        ge=10,
    )

    maximum_heading_words: int = Field(
        default=14,
        ge=1,
    )

    minimum_heading_alphabetic_characters: int = Field(
        default=4,
        ge=1,
    )

    maximum_heading_terminal_punctuation: int = Field(
        default=1,
        ge=0,
    )

    reject_equation_like_blocks: bool = True
    reject_url_like_blocks: bool = True
    reject_numeric_table_fragments: bool = True

    minimum_parent_characters: int = Field(
        default=20,
        ge=1,
    )

    merge_small_structural_groups: bool = True

    preserve_raw_text_when_identical: bool = True