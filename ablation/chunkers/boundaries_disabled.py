"""A6: boundaries-disabled chunker.

DESIGN NOTE: This chunker intentionally violates the SAH-RAG invariant that
child chunks must be contained within their parent's span. To do so, it
returns plain dicts (matching the chunks.jsonl schema) instead of
ChildChunk objects, since ChildChunk enforces the containment invariant.

Each chunk is assigned to the parent with maximum character overlap, so
downstream aggregation still works, but the chunk's span can extend beyond
that parent's span.
"""
from collections import defaultdict

from sah_rag.chunking.windows import build_token_windows
from sah_rag.chunking.config import ChildChunkerConfig

try:
    from sah_rag.artifacts.identity import build_chunk_id
except ImportError:
    from sah_rag.artifacts.fingerprints import build_chunk_id


class BoundariesDisabledChunker:
    CHUNKER_ID = "ablation_boundaries_disabled"
    CHUNKER_VERSION = "0.1.0"

    def __init__(self, *, tokenizer, config: ChildChunkerConfig | None = None):
        self.tokenizer = tokenizer
        if config is None:
            config = ChildChunkerConfig(
                chunker_id=self.CHUNKER_ID,
                chunker_version=self.CHUNKER_VERSION,
            )
        else:
            config = config.model_copy(update={
                "chunker_id": self.CHUNKER_ID,
                "chunker_version": self.CHUNKER_VERSION,
            })
        self.config = config
        # Compute derived values
        self.content_chunk_size = config.content_chunk_size(
            special_tokens_count=tokenizer.special_tokens_count
        )
        self.minimum_final_chunk_tokens = config.effective_minimum_final_chunk_tokens(
            content_chunk_size=self.content_chunk_size
        )
        # Deterministic config hash for chunk_id
        import hashlib, json as _json
        payload = _json.dumps({
            "chunker_id": config.chunker_id,
            "chunker_version": config.chunker_version,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "minimum_final_chunk_tokens": self.minimum_final_chunk_tokens,
            "tokenizer_id": tokenizer.tokenizer_id,
        }, sort_keys=True).encode()
        self.config_hash = hashlib.sha256(payload).hexdigest()

    def chunk_document_with_doc_text(self, canonical_document, parsed_document):
        """Return a tuple of dicts matching the chunks.jsonl schema.

        Each dict has: chunk_id, parent_id, doc_id, dataset, text_normalized,
        chunk_index, parent_chunk_index, token_count.
        """
        text = canonical_document.text_normalized
        parents = parsed_document.parents
        if not parents or not text:
            return ()

        tokens = self.tokenizer.tokenize_with_offsets(text)
        if not tokens:
            return ()

        windows = build_token_windows(
            token_count=len(tokens),
            chunk_size=self.content_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            minimum_final_chunk_tokens=self.minimum_final_chunk_tokens,
        )

        per_parent_count = defaultdict(int)
        chunks = []
        for global_idx, tw in enumerate(windows):
            window_tokens = tokens[tw.start:tw.end]
            cs = window_tokens[0].start
            ce = window_tokens[-1].end
            chunk_text = text[cs:ce]

            # Find parent with maximum span overlap
            best_parent = parents[0]
            best_overlap = -1
            for p in parents:
                ps, pe = p.normalized_char_span.start, p.normalized_char_span.end
                ov = max(0, min(ce, pe) - max(cs, ps))
                if ov > best_overlap:
                    best_overlap = ov
                    best_parent = p

            per_parent_idx = per_parent_count[best_parent.parent_id]
            per_parent_count[best_parent.parent_id] += 1

            cid = build_chunk_id(
                dataset=best_parent.dataset,
                doc_id=best_parent.doc_id,
                parent_id=best_parent.parent_id,
                chunker_config_hash=self.config_hash,
                parent_chunk_index=per_parent_idx,
            )

            chunks.append({
                "chunk_id": cid,
                "parent_id": best_parent.parent_id,
                "doc_id": best_parent.doc_id,
                "dataset": best_parent.dataset,
                "text_normalized": chunk_text,
                "chunk_index": global_idx,
                "parent_chunk_index": per_parent_idx,
                "token_count": tw.end - tw.start,
            })

        return tuple(chunks)