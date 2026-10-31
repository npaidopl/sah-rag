"""Build a LongRAG-style long context from retrieved chunks."""

from __future__ import annotations

from baselines.base import RetrievedChunk

_DEFAULT_SEPARATOR = "\n\n---\n\n"
_DEFAULT_MAX_CHARS = 120_000  # ~30K tokens, matching the LongRAG paper context


def build_context(
    chunks: list[RetrievedChunk],
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    separator: str = _DEFAULT_SEPARATOR,
) -> tuple[str, int]:
    """Concatenate unique chunk texts into a single context string.

    Returns (context, num_unique_units).

    Deduplication is by exact text. In LongRAG, a single long retrieval unit
    expands to multiple document-level RetrievedChunks that all share the same
    unit text, so dedupe prevents multi-counting the same unit in the context.

    Chunks are processed in retrieval order (rank ascending). Concatenation
    stops before ``max_chars`` would be exceeded. If the first chunk alone
    exceeds ``max_chars``, it is truncated to ``max_chars`` so the context is
    never empty for a non-empty input.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if not chunks:
        return "", 0

    seen: set[str] = set()
    parts: list[str] = []
    used = 0
    sep_len = len(separator)

    for chunk in chunks:
        text = chunk.text or ""
        if not text or text in seen:
            continue

        add_len = len(text) + (sep_len if parts else 0)
        if used + add_len > max_chars:
            if not parts:
                parts.append(text[:max_chars])
                used = max_chars
            break

        seen.add(text)
        parts.append(text)
        used += add_len

    return separator.join(parts), len(parts)