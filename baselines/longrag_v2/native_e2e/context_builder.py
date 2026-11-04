"""Build a LongRAG-style long context + title list from retrieved chunks.

The official LongRAG reader prompts (nq, hotpotqa) require BOTH:
    {context} : concatenated document texts
    {titles}  : the list of Wikipedia document titles, formatted as
                ['"t1"', '"t2"', ...]

Both are produced together so the runner receives a coherent unit.
"""

from __future__ import annotations

from dataclasses import dataclass

from baselines.base import RetrievedChunk

_DEFAULT_SEPARATOR = "\n\n---\n\n"
_DEFAULT_MAX_CHARS = 120_000  # ~30K tokens, matching the LongRAG paper context


@dataclass(frozen=True)
class BuiltContext:
    context: str
    titles: list[str]
    num_unique_units: int


def _title_for(chunk: RetrievedChunk) -> str:
    """Best available title for a chunk.

    Resolution order:
        1. chunk.metadata["title"] if present and non-empty
        2. chunk.doc_id (always present, non-empty)

    The LongRAG canonical datasets do not currently carry per-document
    titles in metadata, so doc_id is the practical fallback. This is
    faithful enough for the reader prompt: the official format only
    requires a stable identifier per document.
    """
    md = chunk.metadata or {}
    t = md.get("title")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return chunk.doc_id


def build_context(
    chunks: list[RetrievedChunk],
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    separator: str = _DEFAULT_SEPARATOR,
) -> BuiltContext:
    """Concatenate unique chunk texts and collect their titles.

    Dedupe is by exact text: in LongRAG, one unit expands to multiple
    document-level chunks sharing the same unit text, so dedupe prevents
    multi-counting.

    Titles are collected in retrieval order (rank ascending), one per
    unique chunk retained. If a chunk is dropped by dedupe or by the
    char budget, its title is not included.

    Truncation: if the first chunk exceeds ``max_chars``, it is
    truncated so the context is never empty for a non-empty input.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if not chunks:
        return BuiltContext(context="", titles=[], num_unique_units=0)

    seen_texts: set[str] = set()
    seen_titles: set[str] = set()
    parts: list[str] = []
    titles: list[str] = []
    used = 0
    sep_len = len(separator)

    for chunk in chunks:
        text = chunk.text or ""
        if not text or text in seen_texts:
            continue

        add_len = len(text) + (sep_len if parts else 0)
        if used + add_len > max_chars:
            if not parts:
                parts.append(text[:max_chars])
                used = max_chars
                title = _title_for(chunk)
                if title not in seen_titles:
                    titles.append(title)
                    seen_titles.add(title)
                seen_texts.add(text)
            break

        seen_texts.add(text)
        parts.append(text)
        used += add_len

        title = _title_for(chunk)
        if title not in seen_titles:
            titles.append(title)
            seen_titles.add(title)

    return BuiltContext(
        context=separator.join(parts),
        titles=titles,
        num_unique_units=len(parts),
    )