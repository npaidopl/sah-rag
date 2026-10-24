# Multi-Meta-RAG Baseline Adapter

**Status:** executable (native + controlled)
**Version:** 0.2.0
**Baseline author:** Poliakov & Shvai (ICTERI 2024 Posters)
**Adapter implemented by:** SAH-RAG Project
**Upstream license:** MIT (code) / CC BY 4.0 (paper)

## Overview

Multi-Meta-RAG filters the vector store by metadata extracted from
the query before performing similarity search. This adapter supports
two modes.

### Native mode (paper-exact / author-supported)

- **Extraction:** gpt-3.5-turbo-1106 (paper-native). Requires `OPENAI_API_KEY`.
- **Embedder:** bge-large-en-v1.5. The paper also tests voyage-02, but
  that is API-only. bge-large-en-v1.5 is the author-supported local,
  free alternative.
- **Reranker:** bge-reranker-large (author-supported, optional; if
  weights missing, retrieval proceeds without rerank).
- **Vector store:** in-memory numpy (semantically equivalent to the
  paper's Neo4j $in/$nin metadata filtering).
- **Chunking:** 256 words / 32 overlap (word-based approximation of
  the paper's 256-token / 32-overlap LlamaIndex splitter).

### Controlled mode (LLM-free)

- **Extraction:** deterministic word-boundary substring matching
  against corpus-derived source vocabulary. Justified by the companion
  paper arXiv:2607.03929 (88.0% vs 80.9% set-exact accuracy).
- **Embedder:** injected (BGE-M3) by orchestrator.
- **Reranker:** not applied.

## References

- **Paper:** Poliakov & Shvai, *Multi-Meta-RAG*, ICTERI 2024 Posters. arXiv:2406.13213.
- **Probe companion:** Poliakov & Shvai, *Probe, Don't Prompt*, 2026. arXiv:2607.03929.
- **Repository:** https://github.com/mxpoliakov/Multi-Meta-RAG

## Storage layout

- **Code (this folder):** lives on C:, committed to GitHub.
- **Heavy assets (models, embedders, LLMs):** live on `D:\rag_work\`.
  - Embedder: `D:\rag_work\models\embedders\bge-large-en-v1.5\`
  - Reranker: `D:\rag_work\models\rerankers\bge-reranker-large\`
  - Overridable via `config.extra.*_path` or env vars
    `MULTI_META_RAG_EMBEDDER_PATH` / `MULTI_META_RAG_RERANKER_PATH`.

## Setup for native mode

1. **Download bge-large-en-v1.5** to the embedder path above.
   (from HuggingFace: `BAAI/bge-large-en-v1.5`)
2. **(Optional) Download bge-reranker-large** to the reranker path.
3. **Set API key**: `$env:OPENAI_API_KEY = "sk-..."`.
4. **Install heavy deps** (not required for controlled mode):
   `pip install sentence-transformers FlagEmbedding openai`

The adapter lazy-imports these packages, so a missing dependency only
fails when native mode is actually invoked.

## Supported modes

| Mode | Supported | Paper-exact? | Requires |
|---|---|---|---|
| native | YES | YES (gpt-3.5 + bge-large) | OPENAI_API_KEY + bge-large weights |
| controlled | YES | NO (deterministic) | injected embedder |

## Components (author-declared)

| # | Stage | Original | Native | Controlled |
|---|---|---|---|---|
| 1 | Metadata extraction | GPT-3.5 | GPT-3.5 | substring match |
| 2 | Chunking | 256 tok / 32 | 256 words / 32 | same |
| 3 | Embedding | bge-large / voyage-02 | bge-large | injected BGE-M3 |
| 4 | Vector store | Neo4j | numpy | numpy |
| 5 | Filtering | Neo4j $in / $nin | numpy mask | numpy mask |
| 6 | Candidates | top-20 cosine | top-k*4 (then rerank) | top-k |
| 7 | Reranking | bge-reranker-large | bge-reranker-large | not applied |
| 8 | Generation | GPT-4 / PaLM | out of scope (v2 IR-only) | out of scope |

## Usage

Native:

    from pathlib import Path
    from baselines.base import BaselineConfig, RunMode
    from workspace.multi_meta_rag.adapter import MultiMetaRAGBaseline

    cfg = BaselineConfig(name="multi_meta_rag", mode=RunMode.NATIVE, top_k=10)
    baseline = MultiMetaRAGBaseline(
        config=cfg, work_dir=Path("workspace/_idx/multi_meta_rag_native"),
        embedder=None, llm=None,
    )
    baseline.build_index(documents)
    results = baseline.retrieve("What did The Verge say about AI?")

Controlled:

    cfg = BaselineConfig(name="multi_meta_rag", mode=RunMode.CONTROLLED, top_k=10)
    baseline = MultiMetaRAGBaseline(
        config=cfg, work_dir=Path("workspace/_idx/multi_meta_rag_ctrl"),
        embedder=injected_bge_m3, llm=None,
    )

## Tests

    python -m pytest workspace\multi_meta_rag\tests\ -v

12 tests (6 controlled + 6 native). Offline, mock-based, < 3s on CPU.

## Known limitations

- **Native extraction requires OPENAI_API_KEY.** Without it, native
  mode fails at construction time (fail-fast).
- **Word-based chunking approximation.** Paper uses LlamaIndex token
  splitter; we approximate with word-based to avoid a heavy dep.
- **Reranker optional.** If `bge-reranker-large` weights are missing,
  native retrieval proceeds without rerank (warning suppressed).
- **Expects `metadata['source']`.** Documents without this key
  contribute nothing to the source vocabulary (filtering disabled).
- **`is_llm_based=False`.** Base class uses this flag only for
  controlled-mode validation. Native mode does use an LLM internally
  (documented here and in adapter docstring).

## License notes

Upstream MIT (code) / CC BY 4.0 (paper). This adapter is original
work by the SAH-RAG Project: only the algorithm concept is
reimplemented; no verbatim upstream code is copied.

## Attribution

If you use this adapter in a publication, cite the Multi-Meta-RAG
paper, the probe companion paper, and the SAH-RAG project.