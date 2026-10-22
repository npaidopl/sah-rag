"""First deterministic structural parser."""

from __future__ import annotations

from dataclasses import replace

from sah_rag.artifacts.identity import (
    build_parent_id,
    sha256_fingerprint,
)
from sah_rag.artifacts.models import (
    BoundaryEvidence,
    BoundaryMethod,
    OffsetSpan,
    ParsedDocument,
    SemanticParent,
)
from sah_rag.canonical.models import CanonicalDocument
from sah_rag.parsing.blocks import TextBlock
from sah_rag.parsing.config import (
    DeterministicParserConfig,
)
from sah_rag.parsing.headings import (
    detect_heading_reason,
    is_heading_candidate,
)
from sah_rag.parsing.segmentation import (
    iter_paragraph_spans,
)


class DeterministicStructuralParser:
    """Parse canonical text using observable structural signals."""

    def __init__(
        self,
        config: DeterministicParserConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else DeterministicParserConfig()
        )

        self.config_hash = sha256_fingerprint(
            self.config.model_dump(mode="json")
        )

    def parse(
        self,
        document: CanonicalDocument,
    ) -> ParsedDocument:
        """Parse one canonical document into semantic parents."""

        blocks = self._detect_headings(
            iter_paragraph_spans(
                document.text_normalized
            )
        )

        grouped_blocks = self._group_blocks(blocks)

        parents = tuple(
            self._build_parent(
                document=document,
                parent_index=parent_index,
                blocks=group,
                is_first=parent_index == 0,
                is_last=(
                    parent_index
                    == len(grouped_blocks) - 1
                ),
            )
            for parent_index, group in enumerate(
                grouped_blocks
            )
        )

        return ParsedDocument(
            doc_id=document.doc_id,
            dataset=document.dataset,
            parser_id=self.config.parser_id,
            parser_version=self.config.parser_version,
            parser_config_hash=self.config_hash,
            canonical_text_normalized_length=len(
                document.text_normalized
            ),
            canonical_text_raw_length=len(
                document.text_raw
            ),
            parents=parents,
            source_fingerprint=sha256_fingerprint(
                {
                    "doc_id": document.doc_id,
                    "dataset": document.dataset,
                    "text_raw": document.text_raw,
                    "text_normalized": (
                        document.text_normalized
                    ),
                }
            ),
            metadata={
                "block_count": len(blocks),
                "parent_count": len(parents),
                "parser_mode": "deterministic",
            },
        )

    def _detect_headings(
        self,
        blocks: tuple[TextBlock, ...],
    ) -> tuple[TextBlock, ...]:
        """Annotate blocks that match heading rules."""

        return tuple(
            replace(
                block,
                is_heading=is_heading_candidate(
                    block,
                    self.config,
                ),
            )
            for block in blocks
        )

    def _group_blocks(
        self,
        blocks: tuple[TextBlock, ...],
    ) -> tuple[tuple[TextBlock, ...], ...]:
        """Create sections only from explicit structural headings."""

        if not blocks:
            return ()

        heading_indices = [
            index
            for index, block in enumerate(blocks)
            if block.is_heading
        ]

        if not heading_indices:
            return (blocks,)

        groups: list[tuple[TextBlock, ...]] = []

        first_heading_index = heading_indices[0]

        if first_heading_index > 0:
            preamble = blocks[:first_heading_index]

            if preamble:
                groups.append(preamble)

        for position, heading_index in enumerate(
            heading_indices
        ):
            next_heading_index = (
                heading_indices[position + 1]
                if position + 1 < len(heading_indices)
                else len(blocks)
            )

            group = blocks[
                heading_index:next_heading_index
            ]

            if group:
                groups.append(group)

        grouped = tuple(groups)

        if not self.config.merge_small_structural_groups:
            return grouped

        return self._merge_small_groups(grouped)

    def _merge_small_groups(
        self,
        groups: tuple[tuple[TextBlock, ...], ...],
    ) -> tuple[tuple[TextBlock, ...], ...]:
        """Merge undersized orphan groups without losing offsets."""

        if len(groups) < 2:
            return groups

        mutable_groups = [
            list(group)
            for group in groups
        ]

        index = 0

        while index < len(mutable_groups):
            group = mutable_groups[index]
            group_length = (
                group[-1].end - group[0].start
            )

            if (
                group_length
                >= self.config.minimum_parent_characters
            ):
                index += 1
                continue

            if index + 1 < len(mutable_groups):
                mutable_groups[index + 1] = (
                    group + mutable_groups[index + 1]
                )
                del mutable_groups[index]
                continue

            if index > 0:
                mutable_groups[index - 1].extend(group)
                del mutable_groups[index]
                break

            index += 1

        return tuple(
            tuple(group)
            for group in mutable_groups
            if group
        )

    def _build_parent(
        self,
        *,
        document: CanonicalDocument,
        parent_index: int,
        blocks: tuple[TextBlock, ...],
        is_first: bool,
        is_last: bool,
    ) -> SemanticParent:
        """Build one validated semantic-parent artifact."""

        start = blocks[0].start
        end = blocks[-1].end

        text_normalized = (
            document.text_normalized[start:end]
        )

        heading_block = (
            blocks[0]
            if blocks[0].is_heading
            else None
        )

        heading_reason = (
            detect_heading_reason(
                heading_block,
                self.config,
            )
            if heading_block is not None
            else None
        )

        title = (
            heading_block.text
            if heading_block is not None
            else None
        )

        if is_first:
            start_method = BoundaryMethod.DOCUMENT_START
            start_confidence: float | None = 1.0
        elif heading_block is not None:
            start_method = BoundaryMethod.EXPLICIT_HEADING
            start_confidence = 1.0
        else:
            start_method = BoundaryMethod.DOCUMENT_PREAMBLE
            start_confidence = None

        start_boundary = (
            BoundaryEvidence(
                method=start_method,
                confidence=start_confidence,
            ),
        )

        end_boundary = (
            BoundaryEvidence(
                method=(
                    BoundaryMethod.DOCUMENT_END
                    if is_last
                    else BoundaryMethod.PARAGRAPH_BREAK
                ),
                confidence=1.0 if is_last else None,
            ),
        )

        raw_span: OffsetSpan | None = None
        text_raw: str | None = None

        if (
            self.config.preserve_raw_text_when_identical
            and document.text_raw
            == document.text_normalized
        ):
            raw_span = OffsetSpan(
                start=start,
                end=end,
            )
            text_raw = document.text_raw[start:end]

        parent_id = build_parent_id(
            dataset=document.dataset,
            doc_id=document.doc_id,
            parser_config_hash=self.config_hash,
            parent_index=parent_index,
        )

        return SemanticParent(
            parent_id=parent_id,
            doc_id=document.doc_id,
            dataset=document.dataset,
            parent_parent_id=None,
            hierarchy_level=0,
            sibling_index=parent_index,
            title=title,
            text_normalized=text_normalized,
            text_raw=text_raw,
            normalized_char_span=OffsetSpan(
                start=start,
                end=end,
            ),
            raw_char_span=raw_span,
            token_span=None,
            token_count=None,
            start_boundary=start_boundary,
            end_boundary=end_boundary,
            parser_id=self.config.parser_id,
            parser_version=self.config.parser_version,
            parser_config_hash=self.config_hash,
            metadata={
                "block_indices": [
                    block.block_index
                    for block in blocks
                ],
                "heading_detected": (
                    heading_block is not None
                ),

                "heading_reason": (
                    heading_reason.value
                    if heading_reason is not None
                    else None
                ),
            },
        )