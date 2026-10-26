"""MemoRAG baseline adapter for SAH-RAG v2.

Architecture-defining component
--------------------------------
Memory-conditioned clue generation prior to retrieval is part of the
MemoRAG architecture. Removing the clue stage would produce a different
architecture, not a weaker MemoRAG. The adapter therefore preserves
long-range memory construction, query-conditioned clue generation,
clue-conditioned retrieval, and (in native end-to-end mode) a downstream
answer path.

References
----------
* Paper: MemoRAG, TheWebConf 2025, arXiv:2409.05591.
* Code:  https://github.com/qhjqhj00/MemoRAG (Apache-2.0).
* Weights: TommyChien/memorag-qwen2-7b-inst (official release).
"""

__version__ = "0.1.0"
__all__: list[str] = []