"""Parent-aware deterministic child chunking."""

from __future__ import annotations

from rag_system.artifacts.identity import (
    build_chunk_id,
    sha256_fingerprint,
)
from rag_system.artifacts.models import (
    ChildChunk,
    OffsetSpan,
    ParsedDocument,
    SemanticParent,
)
from rag_system.chunking.config import (
    ChildChunkerConfig,
)
from rag_system.chunking.tokenization import (
    OffsetTokenizer,
    TokenSpan,
)
from rag_system.chunking.windows import build_token_windows


class HierarchicalChildChunker:
    """Create overlapping child chunks inside semantic parents."""

    def __init__(
        self,
        *,
        tokenizer: OffsetTokenizer,
        config: ChildChunkerConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = (
            config
            if config is not None
            else ChildChunkerConfig()
        )

        self.content_chunk_size = (
            self.config.content_chunk_size(
                special_tokens_count=(
                    self.tokenizer.special_tokens_count
                )
            )
        )

        self.minimum_final_chunk_tokens = (
            self.config
            .effective_minimum_final_chunk_tokens(
                content_chunk_size=(
                    self.content_chunk_size
                )
            )
        )

        self.config_hash = sha256_fingerprint(
            {
                **self.config.model_dump(mode="json"),
                "tokenizer_id": self.tokenizer.tokenizer_id,
                "special_tokens_count": (
                    self.tokenizer.special_tokens_count
                ),
                "content_chunk_size": (
                    self.content_chunk_size
                ),
            }
        )

    def chunk_document(
        self,
        parsed_document: ParsedDocument,
    ) -> tuple[ChildChunk, ...]:
        """Chunk every semantic parent without crossing boundaries."""

        chunks: list[ChildChunk] = []
        global_chunk_index = 0

        for parent in parsed_document.parents:
            parent_chunks = self._chunk_parent(
                parent=parent,
                global_start_index=global_chunk_index,
            )

            chunks.extend(parent_chunks)
            global_chunk_index += len(parent_chunks)

        self._validate_document_chunks(
            parsed_document=parsed_document,
            chunks=chunks,
        )

        return tuple(chunks)

    def _chunk_parent(
        self,
        *,
        parent: SemanticParent,
        global_start_index: int,
    ) -> tuple[ChildChunk, ...]:
        """Create child chunks inside one semantic parent."""

        tokens = self.tokenizer.tokenize_with_offsets(
            parent.text_normalized
        )

        if not tokens:
            if self.config.include_empty_parents:
                return (
                    self._build_empty_chunk(
                        parent=parent,
                        chunk_index=global_start_index,
                    ),
                )

            return ()

        windows = build_token_windows(
            token_count=len(tokens),
            chunk_size=self.content_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            minimum_final_chunk_tokens=(
                self.minimum_final_chunk_tokens
            ),
        )


        chunks: list[ChildChunk] = []

        for parent_chunk_index, token_window in enumerate(
            windows
        ):
            window_tokens = tokens[
                token_window.start:token_window.end
            ]

            chunks.append(
                self._build_chunk(
                    parent=parent,
                    window=window_tokens,
                    chunk_index=(
                        global_start_index
                        + parent_chunk_index
                    ),
                    parent_chunk_index=(
                        parent_chunk_index
                    ),
                    token_start=token_window.start,
                    token_end=token_window.end,
                    total_token_count=len(tokens),
                )
            )

        return tuple(chunks)

    def _build_chunk(
        self,
        *,
        parent: SemanticParent,
        window: tuple[TokenSpan, ...],
        chunk_index: int,
        parent_chunk_index: int,
        token_start: int,
        token_end: int,
        total_token_count: int,
    ) -> ChildChunk:
        """Build one validated child chunk."""

        relative_start = window[0].start
        relative_end = window[-1].end

        absolute_start = (
            parent.normalized_char_span.start
            + relative_start
        )
        absolute_end = (
            parent.normalized_char_span.start
            + relative_end
        )

        text_normalized = parent.text_normalized[
            relative_start:relative_end
        ]

        raw_char_span: OffsetSpan | None = None
        text_raw: str | None = None

        if (
            parent.raw_char_span is not None
            and parent.text_raw is not None
        ):
            raw_start = (
                parent.raw_char_span.start
                + relative_start
            )
            raw_end = (
                parent.raw_char_span.start
                + relative_end
            )

            raw_char_span = OffsetSpan(
                start=raw_start,
                end=raw_end,
            )

            text_raw = parent.text_raw[
                relative_start:relative_end
            ]

        chunk_id = build_chunk_id(
            dataset=parent.dataset,
            doc_id=parent.doc_id,
            parent_id=parent.parent_id,
            chunker_config_hash=self.config_hash,
            parent_chunk_index=parent_chunk_index,
        )

        return ChildChunk(
            chunk_id=chunk_id,
            parent_id=parent.parent_id,
            doc_id=parent.doc_id,
            dataset=parent.dataset,
            chunk_index=chunk_index,
            parent_chunk_index=parent_chunk_index,
            text_normalized=text_normalized,
            text_raw=text_raw,
            normalized_char_span=OffsetSpan(
                start=absolute_start,
                end=absolute_end,
            ),
            parent_normalized_char_span=(
                parent.normalized_char_span
            ),
            raw_char_span=raw_char_span,
            token_span=OffsetSpan(
                start=token_start,
                end=token_end,
            ),
            token_count=token_end - token_start,
            parser_id=parent.parser_id,
            parser_version=parent.parser_version,
            parser_config_hash=(
                parent.parser_config_hash
            ),
            chunker_id=self.config.chunker_id,
            chunker_version=(
                self.config.chunker_version
            ),
            chunker_config_hash=self.config_hash,
            tokenizer_id=self.tokenizer.tokenizer_id,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            metadata={
                "relative_char_start": relative_start,
                "relative_char_end": relative_end,
                "is_first_parent_chunk": (
                    parent_chunk_index == 0
                ),
                "is_final_parent_chunk": (
                    token_end == total_token_count
                ),
                "model_input_token_budget": (
                    self.config.chunk_size
                ),
                "content_token_capacity": (
                    self.content_chunk_size
                ),
                "special_tokens_count": (
                    self.tokenizer.special_tokens_count
                ),
                "estimated_model_input_tokens": (
                    token_end
                    - token_start
                    + self.tokenizer.special_tokens_count
                ),
            },
        )

    def _build_empty_chunk(
        self,
        *,
        parent: SemanticParent,
        chunk_index: int,
    ) -> ChildChunk:
        """Build an explicit zero-token chunk when enabled."""

        chunk_id = build_chunk_id(
            dataset=parent.dataset,
            doc_id=parent.doc_id,
            parent_id=parent.parent_id,
            chunker_config_hash=self.config_hash,
            parent_chunk_index=0,
        )

        start = parent.normalized_char_span.start

        return ChildChunk(
            chunk_id=chunk_id,
            parent_id=parent.parent_id,
            doc_id=parent.doc_id,
            dataset=parent.dataset,
            chunk_index=chunk_index,
            parent_chunk_index=0,
            text_normalized="",
            text_raw="" if parent.text_raw is not None else None,
            normalized_char_span=OffsetSpan(
                start=start,
                end=start,
            ),
            parent_normalized_char_span=(
                parent.normalized_char_span
            ),
            raw_char_span=(
                OffsetSpan(
                    start=parent.raw_char_span.start,
                    end=parent.raw_char_span.start,
                )
                if parent.raw_char_span is not None
                else None
            ),
            token_span=OffsetSpan(
                start=0,
                end=0,
            ),
            token_count=0,
            parser_id=parent.parser_id,
            parser_version=parent.parser_version,
            parser_config_hash=parent.parser_config_hash,
            chunker_id=self.config.chunker_id,
            chunker_version=self.config.chunker_version,
            chunker_config_hash=self.config_hash,
            tokenizer_id=self.tokenizer.tokenizer_id,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            metadata={
                "empty_parent": True,
                "is_first_parent_chunk": True,
                "is_final_parent_chunk": True,
            },
        )

    def _validate_document_chunks(
        self,
        *,
        parsed_document: ParsedDocument,
        chunks: list[ChildChunk],
    ) -> None:
        """Check uniqueness and source relationships."""

        chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Chunking produced duplicate chunk IDs."
            )

        known_parent_ids = {
            parent.parent_id
            for parent in parsed_document.parents
        }

        for expected_index, chunk in enumerate(chunks):
            if chunk.chunk_index != expected_index:
                raise ValueError(
                    "Global chunk indexes are not contiguous."
                )

            if chunk.parent_id not in known_parent_ids:
                raise ValueError(
                    "Chunk references an unknown parent."
                )

            if chunk.doc_id != parsed_document.doc_id:
                raise ValueError(
                    "Chunk references a different document."
                )

            if chunk.dataset != parsed_document.dataset:
                raise ValueError(
                    "Chunk references a different dataset."
                )
