from sah_rag.parsing.deterministic import DeterministicStructuralParser
from sah_rag.chunking.hierarchical import HierarchicalChildChunker
from sah_rag.chunking.config import ChildChunkerConfig
from sah_rag.chunking.tokenization import RegexOffsetTokenizer

from ablation.loaders import iter_canonical_documents_local
from ablation.chunkers.variants import (
    FlatHybridParser, ParagraphParentsParser, PlaceboBoundariesParser,
)
from ablation.chunkers.boundaries_disabled import BoundariesDisabledChunker


def main():
    dataset = "greek_bar_bench"
    N = 3
    print(f"Loading {N} docs from {dataset} ...")
    docs = []
    for doc in iter_canonical_documents_local(dataset):
        docs.append(doc)
        if len(docs) >= N:
            break
    print(f"Loaded {len(docs)} docs:")
    for d in docs:
        print(f"  {d.doc_id}  len={len(d.text_normalized)} chars")
    print()

    tok = RegexOffsetTokenizer()
    cfg = ChildChunkerConfig(chunk_size=256, chunk_overlap=32)
    std = HierarchicalChildChunker(tokenizer=tok, config=cfg)
    a6 = BoundariesDisabledChunker(tokenizer=tok, config=cfg)

    variants = {
        "A4_full_sah":             ("std", DeterministicStructuralParser()),
        "A1_flat_hybrid":          ("std", FlatHybridParser()),
        "A7_paragraph_parents":    ("std", ParagraphParentsParser()),
        "A9_placebo_boundaries":   ("std", PlaceboBoundariesParser(seed=42)),
        "A6_boundaries_disabled":  ("a6",  DeterministicStructuralParser()),
    }

    print(f"{'variant':<26} {'doc':<28} {'#par':>5} {'#chk':>5} {'avg_tok':>8} {'cross':>6}")
    print("-" * 86)
    for vname, (mode, parser) in variants.items():
        for doc in docs:
            parsed = parser.parse(doc)
            if mode == "std":
                chunks = std.chunk_document(parsed)
                cross = 0
            else:
                chunks = a6.chunk_document_with_doc_text(doc, parsed)
                cross = sum(1 for c in chunks if c.metadata.get("a6_crosses_parent_boundary"))
            avg = (sum(c.token_count for c in chunks) / len(chunks)) if chunks else 0
            print(f"{vname:<26} {doc.doc_id:<28} {len(parsed.parents):>5} {len(chunks):>5} {avg:>8.1f} {cross:>6}")

    print()
    print("=== CHUNK_ID UNIQUENESS ACROSS VARIANTS (doc[0]) ===")
    doc = docs[0]
    ids = set()
    collisions = 0
    for vname, (mode, parser) in variants.items():
        parsed = parser.parse(doc)
        cs = std.chunk_document(parsed) if mode == "std" else a6.chunk_document_with_doc_text(doc, parsed)
        for c in cs:
            if c.chunk_id in ids:
                collisions += 1
                print(f"  COLLISION: {vname}: {c.chunk_id}")
            ids.add(c.chunk_id)
    print(f"  total unique chunk_ids: {len(ids)}, collisions: {collisions}")

    print()
    print("=== A6 CROSS-BOUNDARY EXAMPLE ===")
    parsed = variants["A6_boundaries_disabled"][1].parse(doc)
    chunks = a6.chunk_document_with_doc_text(doc, parsed)
    for c in chunks:
        if c.metadata.get("a6_crosses_parent_boundary"):
            print(f"  chunk_id={c.chunk_id}")
            print(f"    span=({c.normalized_char_span.start},{c.normalized_char_span.end})")
            print(f"    parent_span=({c.parent_normalized_char_span.start},{c.parent_normalized_char_span.end})")
            print(f"    parent_id={c.parent_id}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())