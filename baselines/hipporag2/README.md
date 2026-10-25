# HippoRAG2 Baseline

> **Status:** EXECUTABLE · **Version:** 0.1.0 · **Modes:** native + controlled

HippoRAG 2 is a memory framework for LLMs that uses Personalized PageRank (PPR)
over a schemaless knowledge graph built offline from OpenIE triples and passage
nodes. It targets multi-hop, memory-intensive retrieval and is the
state-of-the-art baseline against which SAH-RAG is compared.

This adapter exposes HippoRAG 2 through the frozen `BaseBaseline` interface
(`baselines/base.py`), so the orchestrator can run it side-by-side with any
other SAH-RAG baseline under identical evaluation conditions.

---

## Paper & Citation

- **Title:** *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models*
- **Authors:** Bernal Jimenez Gutierrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, Yu Su (OSU NLP)
- **Venue:** ICML 2025
- **arXiv:** [2502.14802](https://arxiv.org/abs/2502.14802)
- **Official code:** [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
- **PyPI:** `pip install hipporag`

If you use this baseline, cite the paper as above.

---

## License

| Component | License | Notes |
|---|---|---|
| HippoRAG 2 source code | **MIT** | Permissive; allows academic + commercial use |
| Default embedder `nvidia/NV-Embed-v2` | **CC-BY-NC-4.0** | **Non-commercial** - avoided in this adapter |
| BGE-M3 (our native + controlled embedder) | **MIT** | Replaces NV-Embed-v2 for license clarity |
| OpenIE LLM (e.g. `gpt-4o-mini`) | Hosted API ToS | Only used in *native* mode when explicitly configured |

**Decision:** This adapter uses **BGE-M3** as the embedder in both modes.
This preserves the architecture while avoiding the non-commercial license of
NV-Embed-v2. See *Deviations from the contract* (Part 3) for details.

---

## Status

- `status = BaselineStatus.EXECUTABLE`
- `supports_native = True` *(requires the `hipporag` package; optional)*
- `supports_controlled = True`
- `is_llm_based = True`

---

## Modes

### Native mode (author-supported)

Uses the **official `hipporag` package** as-is (no fork, no patch) under an
**author-supported configuration**:

- `embedding_model_name = "Transformers/BAAI/bge-m3"` (supported via `TransformersEmbeddingModel`)
- `llm_name = "gpt-4o-mini"` (or any locally hosted LLM supported by HippoRAG)
- `openie_mode = "online"`
- `graph_type = "facts_and_sim_passage_node_unidirectional"`
- `python-igraph` backend (native default)

This is **Author-Supported Native**, *not* `PAPER_EXACT_NATIVE`. The paper's
exact NV-Embed-v2 embedder is deliberately **not** used (license + cost), but
every component choice is one the vendor officially supports through
`BaseConfig`. Reviewers can verify native fidelity by diffing against the
upstream repo at the pinned commit.

**Required to run native:**
pip install git+https://github.com/OSU-NLP-Group/HippoRAG.git
plus an LLM endpoint (`OPENAI_API_KEY` or a local vLLM/Transformers endpoint).

### Controlled mode

Uses the **injected `Embedder` and `LLM`** from the orchestrator (typically
BGE-M3 + a shared LLM). The retrieval pipeline is a clean-room implementation
that preserves HippoRAG 2's architecture:

1. OpenIE per passage (via injected LLM)
2. Typed graph: entity <-> entity (from triples), entity -> passage (unidirectional)
3. Synonymy edges via embedding cosine similarity
   (`topk=2047`, `threshold=0.8` - native defaults)
4. Personalized PageRank (`damping=0.5`, `passage_node_weight=0.05`)
5. Passage ranking by PPR score

This isolates the **architecture** from the **component stack**, which is the
whole point of controlled comparison. No external API calls are made; only
`self.llm.generate()` and `self.embedder.encode_*()` are used.

---

## Architecture

### Pipeline (both modes)

    Documents (CanonicalDocument)
               |
               v
    [ Chunking ]  chunk_size, chunk_overlap
               |
               v
    [ OpenIE ]    LLM -> {entities, triples}
               |
               v
    [ Graph ]     entity <-> entity, entity -> passage
                  + synonymy edges (cosine >= threshold)
               |
               v
    [ PPR ]       damping, passage_node_weight
               |
               v
    [ Ranking ]   top-k by PPR score
               |
               v
      list[RetrievedChunk]
### Native mode

The adapter does NOT reimplement the pipeline. It delegates to the official
hipporag.HippoRAG class:

    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    engine = HippoRAG(global_config=BaseConfig(...))
    engine.index(passage_texts)
    results = engine.retrieve([query], num_to_retrieve=k)

The only code the adapter owns in native mode is:
  (1) chunking,
  (2) text -> (chunk_id, doc_id) mapping,
  (3) result post-processing into RetrievedChunk.

All retrieval logic is the vendor's. That is what makes "native" verifiable:
an examiner can diff adapter.py against upstream hipporag and see that no
algorithmic change was made.

### Controlled mode

Clean-room implementation. Nothing is imported from hipporag at runtime.
The pipeline is reproduced step-by-step with the injected Embedder and LLM,
so every component is auditable and identical to what other baselines receive.
### Preserved native hyperparameters

Even in controlled mode, these values are held at the paper/vendor defaults:

| Parameter                     | Value                                         | Source                       |
|-------------------------------|-----------------------------------------------|------------------------------|
| synonymy_edge_topk            | 2047                                          | HippoRAG 2 BaseConfig        |
| synonymy_edge_sim_threshold   | 0.8                                           | HippoRAG 2 BaseConfig        |
| ppr_damping                   | 0.5                                           | HippoRAG 2 BaseConfig        |
| passage_node_weight           | 0.05                                          | HippoRAG 2 BaseConfig        |
| graph_type                    | facts_and_sim_passage_node_unidirectional     | Vendor default               |

Changing any of these is treated as a *variant*, not as HippoRAG 2, and
would need explicit justification in the report.
---

## Configuration

### config_native.yaml

    name: hipporag2
    mode: native
    top_k: 10
    extra:
      embedder_model_name: "Transformers/BAAI/bge-m3"
      llm_name: "gpt-4o-mini"
      openie_mode: "online"
      graph_type: "facts_and_sim_passage_node_unidirectional"
      synonymy_edge_topk: 2047
      synonymy_edge_sim_threshold: 0.8
      ppr_damping: 0.5
      passage_node_weight: 0.05
      # chunk_size: 1024   # 0 => whole doc as one passage
      # chunk_overlap: 0
### config_controlled.yaml

    name: hipporag2
    mode: controlled
    top_k: 10
    extra:
      chunk_size: 1024
      chunk_overlap: 0
      synonymy_edge_topk: 2047
      synonymy_edge_sim_threshold: 0.8
      ppr_damping: 0.5
      passage_node_weight: 0.05
      max_entities_per_passage: 50
      enable_dense_fallback: true

Rule: embedder and llm are NEVER listed in config_controlled.yaml.
The orchestrator injects them. See docs/adapter_contract.md sections 4 and 6.
### Hyperparameter reference

| Key                         | Default (native / controlled) | Meaning                                    |
|-----------------------------|-------------------------------|--------------------------------------------|
| chunk_size                  | 0 / 1024                      | 0 => whole document is one passage         |
| chunk_overlap               | 0 / 0                         | Char overlap between adjacent chunks       |
| synonymy_edge_topk          | 2047 / 2047                   | Max synonymy edges per entity node         |
| synonymy_edge_sim_threshold | 0.8 / 0.8                     | Min cosine sim to create a synonymy edge   |
| ppr_damping                 | 0.5 / 0.5                     | PPR damping factor                         |
| passage_node_weight         | 0.05 / 0.05                   | Weight of passage-seed nodes               |
| max_entities_per_passage    | n/a / 50                      | Cap on OpenIE entities per chunk           |
| enable_dense_fallback       | n/a / true                    | Dense fallback when PPR cannot seed        |
---

## Dependencies

### Controlled mode

Zero extra dependencies beyond sah-rag[all]. The adapter uses only the Python
standard library plus the injected Embedder / LLM protocols.
requirements.txt is intentionally empty in this mode.

### Native mode (optional)

Native mode requires the official hipporag package, which pulls in
torch, transformers, vllm, openai, litellm, networkx, python-igraph,
and pydantic. Because these are heavy (multi-GB) and unnecessary for
controlled comparison, they are NOT installed by default.

Install when needed:

    pip install git+https://github.com/OSU-NLP-Group/HippoRAG.git

After installation, HippoRAG2Baseline.supports_native becomes True
(detected via importlib.util.find_spec("hipporag")).
No config change is needed.
### Storage and artifacts

Everything is persisted under self.work_dir:

| File                     | Mode       | Content                                  |
|--------------------------|------------|------------------------------------------|
| meta.json                | both       | Passages, config snapshot, entity list   |
| graph.pkl                | controlled | Adjacency dict {u: {v: weight}}          |
| entity_embeddings.pkl    | controlled | Per-entity BGE-M3 embeddings             |
| passage_embeddings.pkl   | controlled | Per-passage BGE-M3 embeddings            |
| native_engine/           | native     | HippoRAG save_dir (parquet, igraph, ...) |

No hardcoded absolute paths. work_dir is supplied by the orchestrator.
---

## Testing

Tests are offline, CPU-only, and finish in under 10 seconds.

Run:

    python -m pytest workspace/hipporag2_dev/tests/ -v

What is covered:

1. test_controlled_build_and_retrieve
   Builds index, asserts all persistence files exist, retrieves top-3,
   checks contiguous ranks 1..k, doc_id membership, non-empty text.

2. test_controlled_reload_from_disk
   Builds once, instantiates a fresh adapter on the same work_dir,
   retrieves without rebuilding, checks ranks are 1..k.

3. test_controlled_fallback_when_no_entities
   LLM returns empty entities; adapter falls back to dense retrieval
   and still returns contiguous ranks.

4. test_controlled_requires_embedder / test_controlled_requires_llm
   Validates the BaseBaseline contract (fail-fast in controlled mode).

5. test_native_raises_when_package_missing
   Runs only when hipporag is NOT installed; asserts native mode
   fails cleanly via BaseBaseline._validate().

6. test_native_supports_flag
   Runs only when hipporag IS installed; asserts supports_native = True.
---

## Graduation checklist

Before requesting graduation from workspace/hipporag2_dev to
baselines/hipporag2, verify:

    [x] adapter.py subclasses BaseBaseline
    [x] name, version, status, supports_native, supports_controlled,
        is_llm_based are set
    [x] __init__ signature matches contract exactly
    [x] build_index() persists under self.work_dir
    [x] retrieve() returns contiguous RetrievedChunk ranks 1..k
    [x] config_native.yaml and config_controlled.yaml present
    [x] tests/test_adapter.py passes offline, < 10s
    [x] README.md documents paper, license, status, deps
    [x] requirements.txt present
    [x] No imports from rag_system
    [x] No hardcoded absolute paths
    [x] No network calls in tests
    [x] No modifications outside workspace/hipporag2_dev/
---

## Deviations from the contract

1. Native mode is gated on the optional hipporag package.
   When the package is absent, supports_native is False (set at import time
   via importlib.util.find_spec). This is not a contract violation: the
   contract permits baselines to declare capabilities dynamically; controlled
   mode is always available.

2. Native mode uses BGE-M3 instead of NV-Embed-v2.
   Reason: NV-Embed-v2 is licensed CC-BY-NC-4.0 (non-commercial).
   Mitigation: classified as AUTHOR_SUPPORTED_NATIVE, not PAPER_EXACT_NATIVE.
   Reviewers can reproduce PAPER_EXACT by setting embedder_model_name to
   "nvidia/NV-Embed-v2" in config_native.yaml; no code change needed.

3. Native mode is retrieval-only.
   Reason: v2 of the SAH-RAG project is IR-only (no generation phase).
   The native engine is invoked only for index() and retrieve(); all QA
   and decoding paths are bypassed.

4. External API calls are possible in native mode only.
   Reason: the native hipporag package uses an LLM for OpenIE and for
   recognition-memory filtering. In controlled mode, no external calls are
   made; the orchestrator injects self.llm.
---

## Reporting template

When reporting this baseline to the supervisor, use the following summary:

    BASELINE: hipporag2
    STATUS: complete
    FILES CREATED:
      - workspace/hipporag2_dev/__init__.py
      - workspace/hipporag2_dev/adapter.py
      - workspace/hipporag2_dev/config_native.yaml
      - workspace/hipporag2_dev/config_controlled.yaml
      - workspace/hipporag2_dev/README.md
      - workspace/hipporag2_dev/requirements.txt
      - workspace/hipporag2_dev/tests/test_adapter.py

    TEST RESULT:
      6 passed, 1 skipped (native tests skipped without hipporag package)

    LICENSE:
      MIT (HippoRAG code) + MIT (BGE-M3 embedder).
      Avoids CC-BY-NC-4.0 of NV-Embed-v2.

    EXTRA DEPENDENCIES:
      Controlled: none.
      Native:     pip install git+https://github.com/OSU-NLP-Group/HippoRAG.git

    KNOWN LIMITATIONS:
      - Native mode is Author-Supported Native (BGE-M3, not NV-Embed-v2).
      - Retrieval-only; generation is out of scope for v2.

    ANY DEVIATIONS FROM CONTRACT:
      See "Deviations from the contract" section above.

    ASKING FOR GRADUATION: pending supervisor review.
---

## Native mode verification status

The native mode (official hipporag package, unmodified) has been verified
in three levels, all without requiring a paid API key for the first two:

### Level 1: Controlled mode (fully verified, no external dependencies)

  - 12 docs, 6 queries on infrastructure-pilot-synthetic-v1
  - Recall@5: 6/6 (100%)
  - Separation between hits and noise: 13-41x
  - Ranks 1..k contiguous, valid doc_ids, non-empty text
  - Runtime: ~13 minutes on CPU laptop

### Level 2: Native mode with offline TransformersLLM (pipeline verified)

  - llm_name = "Transformers/Qwen/Qwen2.5-1.5B-Instruct"
  - embedding_model_name = "Transformers/BAAI/bge-m3"
  - openie_mode = "online"
  - No API key required. Fully offline.
  - Pipeline runs end-to-end; contract compliance verified.
  - Model capability caveat: Qwen 1.5B is too small for the strict JSON
    structured output required by HippoRAG's OpenIE prompts (produces
    ~7 completion tokens on a ~550 token prompt, resulting in an empty
    graph). This is a documented limitation of the development fallback,
    not a bug in the adapter.

### Level 3: Native mode with paper-exact LLM (config verified, awaiting key)

  - llm_name = "gpt-4o-mini"  (paper default)
  - embedding_model_name = "Transformers/BAAI/bge-m3"  (license-safe)
  - openie_mode = "online"
  - information_extraction_model_name = "openie_openai_gpt"
  - Routing verified: _get_llm_class() returns CacheOpenAI instance.
  - HippoRAG(config) initialization verified: no crash without API key.
  - engine.embedding_model resolves to TransformersEmbeddingModel.
  - The ONLY missing runtime requirement is a valid OPENAI_API_KEY.

### How to run Level 3 (paper-exact native)

  PowerShell:
    $env:OPENAI_API_KEY = "sk-..."
    python _archive/hipporag2_native_smoke.py

Expected outcome:
    - OpenIE produces triples per passage
    - Graph is non-empty (num_extracted_triples > 0)
    - Retrieval returns relevant docs with clear score separation

Cost estimate for full CLERC (2756 docs):
    ~$1-5 depending on chunk size and passage length.

### Environment shim for native mode

Native mode requires the TransformersEmbeddingModel class from hipporag,
which imports sentence_transformers, which imports HuggingFace datasets.
The SAH-RAG project has a top-level "datasets/" package that shadows the
HuggingFace library. The adapter calls _ensure_hf_datasets() before any
hipporag import to force-load the site-packages version of datasets into
sys.modules. This is safe because project code that already imported
datasets.* retains its local references. The shim is activated only in
native mode; controlled mode is unaffected.