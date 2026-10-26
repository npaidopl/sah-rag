"""RAPTOR baseline development workspace (SAH-RAG v2).

RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
Paper: Sarthi et al., ICLR 2024 — arXiv:2401.18059
Official repository: https://github.com/parthsarthi03/raptor (MIT License)

Development happens here in `baselines/raptor/` and graduates to
`baselines/raptor/` after the supervisor verifies tests pass.

Heavy dependencies (torch, transformers, sentence-transformers, umap-learn)
are imported lazily by the adapter, so importing this package stays cheap.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]