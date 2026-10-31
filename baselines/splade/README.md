# SPLADE Baseline Adapter

## Description

SPLADE (SParse Lexical AnD Expansion) is a state-of-the-art **sparse
retrieval** method. It is **not** a full RAG system: it provides
first-stage retrieval only, with no generation, no answer flow, and no
context construction.

This adapter implements SPLADE in two modes:

- **native**: official SPLADE-v3 (`naver/splade-v3`) via Hugging Face
  `transformers`. Sparse encoding = ReLU -> log1p -> max-pool.
- **controlled**: injected BGE-M3 embedder -> dense vector -> top-M
  sparsification -> inverted index + sparse dot-product scoring.

## Paper / Reference

- SPLADE v1 (SIGIR'21 short): arXiv:2107.05720
- SPLADE v2: arXiv:2109.10086
- SPLADE-v3: arXiv:2403.06789
- Official code: https://github.com/naver/splade
- Weights: https://huggingface.co/naver/splade-v3

## License

**Model weights** `naver/splade-v3` are released under
**CC BY-NC-SA 4.0** (non-commercial, share-alike). The code in the
official repository has its own license (see upstream). This adapter is
for academic research only. Do not use commercially.

## Status

- `status: EXECUTABLE`
- `supports_native: True`
- `supports_controlled: True`
- `is_llm_based: False`

## Dependencies

Native mode requires heavy dependencies (not needed for controlled mode):

- `torch >= 2.0`
- `transformers >= 4.40`
- `numpy >= 1.26`

Controlled mode only needs `numpy` (already transitively available via
`pandas` in the project).

## Modes

### Native mode

- Loads `naver/splade-v3` via `transformers.AutoModelForMaskedLM`.
- Sparse encoding per text: `log1p(relu(logits))` -> max-pool over tokens.
- Declared as **AUTHOR_SUPPORTED_NATIVE** (uses official weights and
  official sparse-expansion logic, but via the standard HF API rather
  than the Hydra training repo).

### Controlled mode

- Uses the injected `Embedder` (BGE-M3) to obtain dense vectors.
- Sparsifies each dense vector to its top-M dimensions
  (`sparse_top_m`, default 100).
- Builds an inverted index and scores queries by sparse dot product.
- Declared as **controlled SPLADE approximation**, not native SPLADE.
- Rationale: the `Embedder` Protocol is dense-only; we cannot obtain
  SPLADE-style learned sparse expansion from BGE-M3 through the
  protocol. Top-M sparsification preserves the inverted-index +
  dot-product retrieval structure while using the common embedder.

## Known limitations

- Native mode requires `torch` + `transformers` + model download.
- Controlled mode is not faithful SPLADE; it is a controlled sparse
  retrieval approximation for fair comparison under a shared embedder.
- Document-level retrieval only: each `CanonicalDocument` is one chunk,
  so `chunk_id == doc_id` in `RetrievedChunk`.
- Index persistence is JSON-based; for large corpora a binary format
  (e.g., `.npz`) would be more efficient.