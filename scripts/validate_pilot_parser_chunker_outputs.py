"""Validate parser/chunker outputs (round trip, invariants, no downstream)."""

from __future__ import annotations

from pathlib import Path

from rag_system.artifacts import ChildChunk, ParsedDocument

OUTPUT_ROOT = Path("outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system")
PARSING_DIR = OUTPUT_ROOT / "parsing"
CHUNKING_DIR = OUTPUT_ROOT / "chunking"


def _read_jsonl(path: Path, model_type):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield model_type.model_validate_json(line)


def main() -> int:
    parsed = list(_read_jsonl(PARSING_DIR / "parsed_documents.jsonl", ParsedDocument))
    chunks = list(_read_jsonl(CHUNKING_DIR / "child_chunks.jsonl", ChildChunk))

    assert len(parsed) == 12, f"parsed={len(parsed)}"
    assert len(chunks) >= 12, f"chunks={len(chunks)}"

    doc_ids = {p.doc_id for p in parsed}
    chunk_doc_ids = {c.doc_id for c in chunks}
    assert doc_ids == chunk_doc_ids, "chunk doc_ids do not match parsed"

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(set(chunk_ids)) == len(chunk_ids), "duplicate chunk_id"

    parent_ids = {p.parent_id for p in parsed for p in p.parents}
    for c in chunks:
        assert c.parent_id in parent_ids, f"unknown parent_id {c.parent_id}"

    # No downstream artifacts
    banned = (
        "embeddings.jsonl", "embedding_records.jsonl",
        "retrieval_cases.jsonl", "ranked_items.jsonl",
        "query_metrics.jsonl", "aggregate_metrics.json",
    )
    for p in OUTPUT_ROOT.rglob("*"):
        if p.is_file() and p.name in banned:
            raise SystemExit(f"downstream artifact present: {p}")

    print("[OK] Pilot parser+chunker outputs")
    print(f"  parsed documents: {len(parsed)}")
    print(f"  child chunks: {len(chunks)}")
    print("  unique chunk ids: True")
    print("  parent ids valid: True")
    print("  document coverage: True")
    print("  no downstream artifacts: True")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
