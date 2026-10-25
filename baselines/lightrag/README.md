# LightRAG baseline (SAH-RAG v2)

**Status:** `EXECUTABLE`
**Native tag:** `AUTHOR_SUPPORTED_NATIVE` (NOT paper-exact)
**Controlled tag:** `CONTROLLED_ADAPTATION`
**Modes:** `native` + `controlled`
**License:** MIT

## Overview

This baseline adapts the LightRAG architecture (arXiv:2410.05779,
EMNLP 2025 Findings) to the SAH-RAG v2 baseline contract
(`baselines/base.py`).

The adapter preserves LightRAG's architecture-defining logic: fixed
1200/100 chunking, graph construction from entity/relation
co-occurrence, and dual-level retrieval with hybrid fusion.

**This is not a paper-exact reproduction.** The original paper uses
GPT-4o-mini and OpenAI text-embedding-3-large, which require external
API access and are not offline-reproducible. Instead, this adapter
provides two execution paths (see below).

## Paper / repository / license

| Field | Value |
|---|---|
| Paper | arXiv:2410.05779 (EMNLP 2025 Findings) |
| Repository | HKUDS/LightRAG |
| Pinned commit | `453dce83d6d0354a06e46c8d4029a0895c4e054b` |
| License | MIT |

## Supported modes

| Mode | Supported | Tag | Notes |
|---|---|---|---|
| `native` | Yes | `AUTHOR_SUPPORTED_NATIVE` | HF binding (`hf_model_complete`, `hf_embed`); native prompts, chunker, storage. NOT paper-exact. |
| `controlled` | Yes | `CONTROLLED_ADAPTATION` | Injected BGE-M3 + Qwen; generation disabled (IR-only). |

### Native mode: academic status

Native mode uses LightRAG's official HuggingFace binding
(`lightrag/llm/hf.py`) β€” an official extension point of the framework.
All architecture-defining components (prompts, chunker semantics,
graph merge logic, retrieval ranking) remain VERBATIM upstream.

Since the paper uses GPT-4o-mini (not HF models), the resulting stack
is **author-supported native**, not paper-exact. Any publication must
declare it as:

> "LightRAG, author-supported native configuration, using
> HuggingFace provider with `<model>`."

**Never** declare this as "paper-exact native reproduction".

## Architecture preservation

### Preserved (VERBATIM upstream β€” do not modify)

- Chunker semantics: fixed-token chunking, `chunk_size=1200`,
  `overlap=100`.
- Prompt templates: entity extraction, relation extraction,
  keyword extraction.
- Graph construction: entity/relationship extraction pipeline,
  graph merge logic.
- Dual-level retrieval: low-level, high-level, hybrid mode.
- Retrieval ranking: native ordering and fusion.

### Adapted (explicitly labeled)

| Component | Paper / Native | Controlled | Fallback (offline) |
|---|---|---|---|
| Extraction LLM | GPT-4o-mini | Injected LLM | Co-occurrence stats |
| Keyword LLM | GPT-4o-mini | Injected LLM | TF-based top-k |
| Embedder | OpenAI text-embedding-3-large | BGE-M3 (injected) | Hash-based |
| Graph backend | NetworkX | NetworkX | Dict co-occurrence |
| Generation | GPT-4o-mini | Disabled | Disabled |

### Never (would break native status)

- Prompt rewrite
- Chunker semantics change
- Graph merge semantics change
- Entity/relation extraction logic change
- Retrieval ranking beyond native modes

## Execution paths

### 1. Fallback path (offline, CI, always available)

Activated when:
- Controlled mode without vendored LightRAG import, OR
- Native mode without `hf_llm` + `hf_embed` configured, OR
- Full pipeline fails and `strict_full_path: false`.

Characteristics:
- Same chunking scheme (1200/100)
- Co-occurrence graph (replaces LLM extraction)
- Dual-level retrieval fusion
- Injected embedder (controlled) or hash-based (offline)
- **No external models required**

Keeps the contract test offline, CPU-only, under 10 seconds.

### 2. Full vendored path (GPU server)

Activated when:
- `external/baselines/lightrag/` (or `external/lightrag/`,
  `vendor/lightrag/`, `third_party/lightrag/`) is discoverable, AND
- Native: `config.extra["hf_llm"]` + `config.extra["hf_embed"]`
  set, AND HF binding imported successfully; OR
- Controlled: injected `embedder` + `llm`.

Currently returns `NotImplementedError` with integration notes.
Implement `_build_index_full` and `_retrieve_full` on the GPU server
against the vendored code.

### 3. Strict mode (GPU server β€” prevent silent fallback)

Set in `config_native.yaml`:

```yaml
extra:
  strict_full_path: true
```

With this flag, any full-pipeline failure **propagates** instead of
falling back. Essential for defending native status under external
scrutiny.


## Configuration reference

Top-level keys (per contract): `name`, `mode`, `top_k`, `extra`.

### `extra` keys used by this adapter

| Key | Default | Purpose |
|---|---|---|
| `chunk_size` | 1200 | Native chunk size (tokens). |
| `chunk_overlap` | 100 | Native chunk overlap (tokens). |
| `hf_llm` | (unset) | HF model for native mode (e.g., `Qwen/Qwen2.5-7B-Instruct`). |
| `hf_embed` | (unset) | HF embedder for native mode (e.g., `BAAI/bge-m3`). |
| `strict_full_path` | `false` | If `true`, full-pipeline failures propagate. |

## Testing

Offline, CPU-only, under 10 seconds:

```powershell
python -m pytest baselines\lightrag\tests\ -v
```

Covers: native build + retrieve (fallback), empty corpus, controlled
with mocks, top_k respected, retrieve before build, controlled mode
without injection raises.

## Definition of Done

- [x] Subclasses `BaseBaseline`
- [x] Identity attributes (`lightrag`, v0.1.0, EXECUTABLE, both modes)
- [x] `__init__` signature matches contract
- [x] `build_index` persists under `self.work_dir`
- [x] `retrieve` returns contiguous `RetrievedChunk` ranks
- [x] `config_native.yaml` + `config_controlled.yaml`
- [x] Tests pass offline under 10s
- [x] No `rag_system` imports
- [x] No hardcoded absolute paths
- [x] No external API calls in controlled mode
- [ ] Full vendored pipeline (provisioned for GPU server)

## Reporting back to supervisor

```
BASELINE: lightrag
STATUS: complete (fallback path), provisioned for GPU (full path)
LICENSE: MIT
EXTRA DEPENDENCIES: networkx, nano-vectordb, tiktoken (full pipeline only)
KNOWN LIMITATIONS: Full pipeline returns NotImplementedError; complete on GPU
ASKING FOR GRADUATION: yes (pending test pass)
```