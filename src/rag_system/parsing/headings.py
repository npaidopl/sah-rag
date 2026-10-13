"""Conservative deterministic heading detection."""

from __future__ import annotations

import re
from enum import StrEnum

from rag_system.parsing.blocks import TextBlock
from rag_system.parsing.config import (
    DeterministicParserConfig,
)

_SECTION_NUMBER = re.compile(
    r"""
    ^
    (?:
        \d+\.
        |
        \d+\.\d+(?:\.\d+)*[.)]?
        |
        \(\d+\)
        |
        [IVXLCM]+[.)]
        |
        [A-ZΑ-Ω][.)]
    )
    \s*
    \S
    """,
    re.VERBOSE,
)

_NAMED_HEADING = re.compile(
    r"""
    ^
    (?:
        chapter
        | section
        | article
        | part
        | title
        | appendix
        | annex
        | exhibit
        | schedule
        | κεφάλαιο
        | κεφαλαιο
        | ενότητα
        | ενοτητα
        | άρθρο
        | αρθρο
        | μέρος
        | μερος
        | παράρτημα
        | παραρτημα
    )
    (?:
        \s+
        [A-Za-zΑ-Ωα-ω0-9IVXLCM.-]+
    )?
    (?:
        \s*[:.-]\s*
        [^.!?]{1,80}
    )?
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

_URL_PATTERN = re.compile(
    r"(?:https?://|www\.|<a\s|href=)",
    re.IGNORECASE,
)

_EQUATION_PATTERN = re.compile(
    r"""
    (?:
        [A-Za-z_]\w*
        \s*[=*/+]\s*
        [A-Za-z0-9_(]
    )
    """,
    re.VERBOSE,
)

_NUMERIC_FRAGMENT = re.compile(
    r"""
    ^
    [\d\s.,%$€£¥+\-–—()/]+
    $
    """,
    re.VERBOSE,
)


class HeadingReason(StrEnum):
    """Reason a block was classified as a heading."""

    NUMBERED = "numbered"
    NAMED = "named"
    UPPERCASE = "uppercase"


def detect_heading_reason(
    block: TextBlock,
    config: DeterministicParserConfig,
) -> HeadingReason | None:
    """Return a conservative heading classification."""

    text = block.text.strip()

    if not _passes_basic_filters(text, config):
        return None

    if (
        config.detect_named_headings
        and _NAMED_HEADING.fullmatch(text)
    ):
        return HeadingReason.NAMED

    if (
        config.detect_numbered_headings
        and _SECTION_NUMBER.search(text)
        and _looks_like_numbered_heading(text)
    ):
        return HeadingReason.NUMBERED

    if (
        config.detect_uppercase_headings
        and _is_uppercase_heading(text, config)
    ):
        return HeadingReason.UPPERCASE

    return None


def is_heading_candidate(
    block: TextBlock,
    config: DeterministicParserConfig,
) -> bool:
    """Return whether a block is an accepted structural heading."""

    return detect_heading_reason(block, config) is not None


def _passes_basic_filters(
    text: str,
    config: DeterministicParserConfig,
) -> bool:
    """Reject blocks that cannot be structural headings."""

    if not text:
        return False

    if "\n" in text or "\r" in text:
        return False

    if len(text) > config.maximum_heading_characters:
        return False

    if len(text.split()) > config.maximum_heading_words:
        return False

    alphabetic_count = sum(
        character.isalpha()
        for character in text
    )

    if (
        alphabetic_count
        < config.minimum_heading_alphabetic_characters
    ):
        return False

    if (
        config.reject_url_like_blocks
        and _URL_PATTERN.search(text)
    ):
        return False

    if (
        config.reject_equation_like_blocks
        and _EQUATION_PATTERN.search(text)
    ):
        return False

    if (
        config.reject_numeric_table_fragments
        and _NUMERIC_FRAGMENT.fullmatch(text)
    ):
        return False

    terminal_punctuation = sum(
        text.count(character)
        for character in ".!?"
    )

    return not terminal_punctuation > config.maximum_heading_terminal_punctuation


def _looks_like_numbered_heading(text: str) -> bool:
    """Reject sentences, years, addresses, and table rows."""

    words = text.split()

    if len(words) < 2:
        return False

    first_token = words[0]

    if first_token.isdigit():
        return False

    remainder = " ".join(words[1:])

    if not any(
        character.isalpha()
        for character in remainder
    ):
        return False

    sentence_like_endings = (
        ",",
        ";",
        "?",
        "!",
    )

    return not text.endswith(sentence_like_endings)


def _is_uppercase_heading(
    text: str,
    config: DeterministicParserConfig,
) -> bool:
    """Detect short uppercase headings while rejecting data rows."""

    if any(
        operator in text
        for operator in ("=", "*", "<", ">", "{", "}")
    ):
        return False

    alphabetic_characters = [
        character
        for character in text
        if character.isalpha()
    ]

    if (
        len(alphabetic_characters)
        < config.minimum_heading_alphabetic_characters
    ):
        return False

    uppercase_count = sum(
        character.isupper()
        for character in alphabetic_characters
    )

    uppercase_ratio = (
        uppercase_count
        / len(alphabetic_characters)
    )

    if uppercase_ratio < 0.9:
        return False

    words = text.split()

    return not (len(words) == 1 and len(text) < 5)