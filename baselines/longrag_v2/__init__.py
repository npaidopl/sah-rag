"""LongRAG baseline adapter for the SAH-RAG v2 project.

Reference:
    Jiang, Z., Ma, X., Chen, W. (2024).
    "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs."
    arXiv:2406.15319. Official repository: https://github.com/TIGER-AI-Lab/LongRAG
    (MIT license).

This adapter implements the retrieval side of LongRAG under the frozen
`BaseBaseline` contract. It does NOT implement the long-context reader
(the reader is out of scope for v2, which is IR-only per the adapter
contract).

Modes:
    native     -- uses the paper's embedder (BGE-large-en-v1.5) and the
                  paper's long-unit grouping logic.
    controlled -- uses the orchestrator-injected embedder (BGE-M3) while
                  preserving LongRAG's architecture-defining retrieval logic
                  (long-unit grouping, top-k1 unit retrieval, unit-to-document
                  expansion via MaxP approximation).
"""

from baselines.longrag_v2.adapter import LongRAGBaseline

__all__ = ["LongRAGBaseline"]
__version__ = "0.1.0"