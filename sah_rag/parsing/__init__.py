"""Document parsing interfaces and implementations."""

from __future__ import annotations

from sah_rag.parsing.blocks import TextBlock
from sah_rag.parsing.config import (
    DeterministicParserConfig,
)
from sah_rag.parsing.deterministic import (
    DeterministicStructuralParser,
)
from sah_rag.parsing.headings import (
    HeadingReason,
    detect_heading_reason,
    is_heading_candidate,
)
from sah_rag.parsing.pipeline import (
    ParsingRunSummary,
    parse_canonical_dataset,
)
from sah_rag.parsing.profiling import (
    ParserProfileAccumulator,
)
from sah_rag.parsing.segmentation import (
    iter_paragraph_spans,
)

__all__ = [
    "HeadingReason",
    "ParserProfileAccumulator",
    "DeterministicParserConfig",
    "DeterministicStructuralParser",
    "ParsingRunSummary",
    "TextBlock",
    "detect_heading_reason",
    "is_heading_candidate",
    "iter_paragraph_spans",
    "parse_canonical_dataset",
]