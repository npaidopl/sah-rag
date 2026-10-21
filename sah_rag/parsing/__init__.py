"""Document parsing interfaces and implementations."""

from __future__ import annotations

from rag_system.parsing.blocks import TextBlock
from rag_system.parsing.config import (
    DeterministicParserConfig,
)
from rag_system.parsing.deterministic import (
    DeterministicStructuralParser,
)
from rag_system.parsing.headings import (
    HeadingReason,
    detect_heading_reason,
    is_heading_candidate,
)
from rag_system.parsing.pipeline import (
    ParsingRunSummary,
    parse_canonical_dataset,
)
from rag_system.parsing.profiling import (
    ParserProfileAccumulator,
)
from rag_system.parsing.segmentation import (
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