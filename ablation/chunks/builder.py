"""Build chunks.jsonl for one (variant, dataset). GT-aware sampling."""
import hashlib
import json
from pathlib import Path

from ablation.config import CHUNKS_DIR, sample_limit, sample_docs_gt_aware
from ablation.loaders import iter_canonical_documents_local
from ablation.chunkers.variants import (
    FlatHybridParser, ParagraphParentsParser, PlaceboBoundariesParser,
)
from ablation.chunkers.boundaries_disabled import BoundariesDisabledChunker
from sah_rag.parsing.deterministic import DeterministicStructuralParser
from sah_rag.chunking.hierarchical import HierarchicalChildChunker
from sah_rag.chunking.config import ChildChunkerConfig
from sah_rag.chunking.tokenization import RegexOffsetTokenizer


_KNOWN_VARIANTS = {"A4_full_sah", "A7_paragraph_parents", "A1_flat_hybrid",
                   "A9_placebo_boundaries", "A6_boundaries_disabled"}


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_parser_chunker(variant: str, cfg: ChildChunkerConfig, tok):
    if variant == "A4_full_sah":
        return DeterministicStructuralParser(), HierarchicalChildChunker(tokenizer=tok, config=cfg), False
    if variant == "A7_paragraph_parents":
        return ParagraphParentsParser(), HierarchicalChildChunker(tokenizer=tok, config=cfg), False
    if variant == "A1_flat_hybrid":
        return FlatHybridParser(), HierarchicalChildChunker(tokenizer=tok, config=cfg), False
    if variant == "A9_placebo_boundaries":
        return PlaceboBoundariesParser(seed=42), HierarchicalChildChunker(tokenizer=tok, config=cfg), False
    if variant == "A6_boundaries_disabled":
        return DeterministicStructuralParser(), BoundariesDisabledChunker(tokenizer=tok, config=cfg), True
    raise ValueError(f"Unknown variant: {variant}")


def build_chunks_for(variant: str, dataset: str, force: bool = False,
                     apply_sampling: bool = True) -> dict:
    if variant not in _KNOWN_VARIANTS:
        return {"status": "skipped", "reason": f"unknown variant {variant}"}

    out_dir = CHUNKS_DIR / variant / dataset
    out_chunks = out_dir / "chunks.jsonl"
    out_manifest = out_dir / "manifest.json"
    if out_chunks.exists() and out_manifest.exists() and not force:
        return {"status": "already_done", "path": str(out_chunks)}

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ChildChunkerConfig(chunk_size=256, chunk_overlap=32)
    tok = RegexOffsetTokenizer()
    parser, chunker, use_a6 = _get_parser_chunker(variant, cfg, tok)

    # GT-aware sampling
    limit = sample_limit(dataset) if apply_sampling else None
    if limit is not None:
        doc_filter = sample_docs_gt_aware(dataset, limit)
    else:
        doc_filter = None

    n_docs = 0
    n_chunks = 0
    n_docs_skipped = 0
    tmp = out_chunks.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for doc in iter_canonical_documents_local(dataset):
            if doc_filter is not None and doc.doc_id not in doc_filter:
                n_docs_skipped += 1
                continue
            n_docs += 1
            try:
                parsed = parser.parse(doc)
                if use_a6:
                    chunks = chunker.chunk_document_with_doc_text(doc, parsed)
                else:
                    chunks = chunker.chunk_document(parsed)
            except Exception as e:
                print(f"  PARSE ERROR {doc.doc_id}: {e}")
                continue
            def _get(c, name):
                return c[name] if isinstance(c, dict) else getattr(c, name)
            for c in chunks:
                n_chunks += 1
                f.write(json.dumps({
                    "chunk_id": _get(c, "chunk_id"),
                    "parent_id": _get(c, "parent_id"),
                    "doc_id": _get(c, "doc_id"),
                    "dataset": _get(c, "dataset"),
                    "text_normalized": _get(c, "text_normalized"),
                    "chunk_index": _get(c, "chunk_index"),
                    "parent_chunk_index": _get(c, "parent_chunk_index"),
                    "token_count": _get(c, "token_count"),
                }) + "\n")

    if n_chunks == 0:
        tmp.unlink()
        return {"status": "error", "reason": "zero chunks produced"}

    tmp.replace(out_chunks)
    manifest = {
        "variant": variant, "dataset": dataset,
        "n_docs": n_docs, "n_docs_skipped": n_docs_skipped,
        "sample_limit": limit,
        "n_chunks": n_chunks,
        "chunks_sha256": _sha256_file(out_chunks),
        "chunks_size_bytes": out_chunks.stat().st_size,
    }
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "done", "n_docs": n_docs, "n_docs_skipped": n_docs_skipped,
            "n_chunks": n_chunks, "path": str(out_chunks),
            "size_mb": round(out_chunks.stat().st_size / 1024 / 1024, 2)}
