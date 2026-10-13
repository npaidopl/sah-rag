"""Run parsing and child chunking on the synthetic fixture."""

from __future__ import annotations

from pathlib import Path

from rag_system.artifacts.jsonl import write_jsonl_atomic
from rag_system.canonical.models import CanonicalDocument
from rag_system.chunking import (
    ChildChunkerConfig,
    HierarchicalChildChunker,
    RegexOffsetTokenizer,
)
from rag_system.parsing import DeterministicStructuralParser

FIXTURE_PATH = Path(
    "tests/fixtures/structured_document.json"
)

OUTPUT_PATH = Path(
    "outputs/chunking_smoke_test.jsonl"
)


def main() -> None:
    """Parse, chunk, validate, and write the fixture."""

    document = CanonicalDocument.model_validate_json(
        FIXTURE_PATH.read_text(encoding="utf-8-sig")
    )

    parser = DeterministicStructuralParser()
    parsed = parser.parse(document)

    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(
            chunk_size=8,
            chunk_overlap=2,
        ),
    )

    chunks = chunker.chunk_document(parsed)

    write_jsonl_atomic(
        OUTPUT_PATH,
        chunks,
        overwrite=True,
    )

    print("[OK] Document:", parsed.doc_id)
    print("[OK] Parents:", len(parsed.parents))
    print("[OK] Chunks:", len(chunks))
    print(
        "[OK] Chunker config hash:",
        chunker.config_hash,
    )

    for chunk in chunks:
        preview = " ".join(
            chunk.text_normalized.split()
        )[:100]

        print(
            "[CHUNK]",
            chunk.chunk_index,
            chunk.parent_id,
            chunk.parent_chunk_index,
            (
                chunk.token_span.start
                if chunk.token_span is not None
                else None
            ),
            (
                chunk.token_span.end
                if chunk.token_span is not None
                else None
            ),
            repr(preview),
        )

    print("[OK] Written:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
