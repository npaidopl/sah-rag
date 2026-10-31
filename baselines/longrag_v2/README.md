# LongRAG baseline adapter (v2)

**Project:** SAH-RAG v2 - IR-only baseline comparison  
**Adapter name:** `longrag`  
**Version:** `0.1.0`  
**Status:** `executable`  
**Modes:** `native` and `controlled`  
**LLM-based:** no

---

## 1. Description

This adapter integrates the **retrieval** side of LongRAG into the SAH-RAG v2
comparison harness. It preserves LongRAG's architecture-defining retrieval
logic - long retrieval units (>4K tokens) and top-k unit retrieval followed
by unit-to-document expansion - while conforming to the frozen
`BaseBaseline` contract.

The **reader** side of LongRAG (long-context answer generation with
GPT-4o or Gemini-1.5-Pro) is intentionally **not implemented**. The
SAH-RAG v2 harness is IR-only (see `docs/adapter_contract.md` section 9
and Hard Rule #6 in `docs/report_llm_organizer.md`). This is consistent
with all other baselines in the project.

---

## 2. Reference

- **Paper:** Jiang, Z., Ma, X., Chen, W. (2024).
  *LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs.*
  arXiv:2406.15319. https://arxiv.org/abs/2406.15319
- **Official repository:** https://github.com/TIGER-AI-Lab/LongRAG

### What is architecture-defining (and therefore preserved)

| Element | LongRAG behavior | Preserved in this adapter |
|---|---|---|
| Retrieval unit granularity | Long units, ~4K tokens (30x short chunks) | Yes, `target_unit_tokens` (default 4096) |
| Unit construction (Wikipedia) | Hyperlink-based document grouping | Yes, `grouping_policy: hyperlink_based` |
| Unit construction (non-Wikipedia) | One document per unit | Yes, `grouping_policy: one_document_per_unit` |
| Retrieval depth | k1 units retrieved, then expanded | Yes, `k1` (default 100) |
| Unit-to-document mapping | MaxP approximation (best score per document) | Yes, implemented in `retrieve()` |
| Reranker | None | Yes, none |

### What is out of scope (and therefore not implemented)

- The long-context reader (GPT-4o / Gemini-1.5-Pro).
- Answer generation, prompts, decoding.
- Answer-level metrics (EM, F1). The v2 harness evaluates retrieval only.

---

## 3. License

- **Code:** MIT (TIGER-AI-Lab/LongRAG). Permits use, modification, and
  redistribution with attribution.
- **Datasets** (when run on paper datasets): NQ - Apache 2.0;
  HotpotQA - CC BY-SA 4.0. Dataset licenses apply to the data, not to
  this adapter. The SAH-RAG canonical dataset suite handles dataset
  licensing independently.

This adapter is a clean re-implementation of LongRAG's retrieval logic
under the SAH-RAG contract. It does not vendor the official repository;
it re-expresses the architecture-defining logic in the project's
`BaseBaseline` shape.

---
## 4. Supported modes

### 4.1 Native mode (`config_native.yaml`)

- Uses the paper's embedder: **BGE-large-en-v1.5** (CLS pooling, L2-normalized).
- Uses the paper's non-Wikipedia grouping default: `one_document_per_unit`.
- No injected components are required from the orchestrator; the adapter
  loads its own embedder lazily on first `build_index()` call.
- Embedder resolution order:
  1. `extra.embedder_local_path` (absolute path to a local snapshot);
  2. environment variable `LONGRAG_EMBEDDER_PATH`;
  3. HuggingFace hub name `BAAI/bge-large-en-v1.5` (requires cache or
     network access).

### 4.2 Controlled mode (`config_controlled.yaml`)

- Uses the **orchestrator-injected embedder** (BGE-M3 in the SAH-RAG
  controlled protocol). The adapter never loads BGE-large-en-v1.5 in
  this mode.
- Uses the grouping policy that matches the existing controlled LongRAG
  pipeline (L1b..L7): `adjacent_document_segments` at
  `target_unit_tokens: 4096`.
- The injected embedder is accessed only through the `Embedder` protocol
  (`encode_documents`, `encode_query`, `name`).
- `llm` is not needed: LongRAG's retrieval pipeline is LLM-free, and
  `is_llm_based = False`.

---

## 5. Grouping policies

The adapter supports three grouping policies, selectable via
`extra.grouping_policy`:

| Policy | When to use | Paper-faithful? |
|---|---|---|
| `one_document_per_unit` | Non-Wikipedia corpora (canonical suite: clerk, cuad, greek_bar_bench, lara). Default in native mode. | Yes, per paper section 2 |
| `adjacent_document_segments` | Heterogeneous corpora where documents are much smaller than 4K tokens. Default in controlled mode. | No (adaptation) |
| `hyperlink_based` | Wikipedia tasks where `metadata["hyperlinks"]` is present on each `CanonicalDocument`. Falls back to `one_document_per_unit` with a warning if no hyperlinks are found. | Yes, per paper section 2 |

Token estimation for grouping uses a cheap heuristic
(`max(whitespace_count, ceil(len/4))`) so that controlled mode stays
tokenizer-agnostic. This is documented in `adapter.py::_estimate_tokens`.

---

## 6. Dependencies

### Base install (always required)

- `numpy` - used by the in-memory retrieval index. Already a transitive
  dependency of `pandas` in the SAH-RAG base install, but listed
  explicitly in `requirements.txt` so this adapter can be validated in
  isolation.

### Native mode only (heavy; opt-in)

- `torch` and `transformers` - needed to load BGE-large-en-v1.5 locally.
- **Not required** for controlled mode.
- Commented out in `requirements.txt` by design; the SAH-RAG supervisor
  decides when to enable them (they are heavy and mostly GPU-relevant).

---

## 7. Configuration schema

Both configs follow the frozen schema from `docs/adapter_contract.md`
section 5:

```yaml
name: longrag
mode: native          # or "controlled"
top_k: 10
extra:
  grouping_policy: one_document_per_unit   # or adjacent_document_segments / hyperlink_based
  target_unit_tokens: 4096
  k1: 100
  embedder_local_path: null                # native only
```

Top-level keys are **only** `name`, `mode`, `top_k`, `extra`. Component
injection in controlled mode is handled by the orchestrator; the
controlled config does **not** list the embedder.

---
## 8. Usage

### 8.1 Native mode

```python
from pathlib import Path
from baselines.base import BaselineConfig, RunMode
from baselines.longrag_v2.adapter import LongRAGBaseline

cfg = BaselineConfig(
    name="longrag",
    mode=RunMode.NATIVE,
    top_k=10,
    extra={
        "grouping_policy": "one_document_per_unit",
        "target_unit_tokens": 4096,
        "k1": 100,
    },
)
baseline = LongRAGBaseline(config=cfg, work_dir=Path("./runs/longrag_native"))
baseline.build_index(corpus)
results = baseline.retrieve("legal indemnity clause", top_k=10)
```

### 8.2 Controlled mode

```python
baseline = LongRAGBaseline(
    config=cfg,                 # mode=RunMode.CONTROLLED
    work_dir=Path("./runs/longrag_controlled"),
    embedder=shared_bge_m3,     # injected by the orchestrator
)
baseline.build_index(corpus)
results = baseline.retrieve("legal indemnity clause", top_k=10)
```

In both modes, `retrieve()` returns `list[RetrievedChunk]` with
contiguous ranks 1..k and deterministic tie-breaking (`score` desc,
then `doc_id` asc).

### 8.3 Persistence

`build_index()` persists three artifacts under `self.work_dir`:

- `unit_vectors.npy` - L2-normalized unit vectors (float32).
- `unit_records.json` - `unit_id`, `text`, `doc_ids`, `token_count`
  per unit.
- `index_meta.json` - config snapshot for provenance.

---

## 9. Testing

All tests are offline, CPU-only, and complete in **< 10 seconds**. Native
mode is only exercised when `torch`, `transformers`, and
`LONGRAG_EMBEDDER_PATH` are all present; otherwise the native test is
skipped via `importlib.util.find_spec` (no actual import at collection
time, so collection stays fast).

Verification:

```powershell
python -m pytest baselines.longrag_v2\tests\ -v
```

Expected: **6 passed, 1 skipped** (native smoke skipped without env var).

Test coverage:

- identity attributes match the contract;
- controlled build + retrieve end-to-end with a mock embedder;
- multi-document unit expansion preserves contiguous ranks;
- retrieve-before-build raises `RuntimeError`;
- unknown grouping policy raises `ValueError`;
- empty corpus raises `ValueError`.

---

## 10. Known limitations

1. **The reader is not implemented.** SAH-RAG v2 is IR-only (contract
   section 9). Answer-level LongRAG reproduction (EM / F1) is out of
   scope for this adapter.
2. **Token estimation is approximate.** `_estimate_tokens` uses a cheap
   heuristic instead of a tokenizer to keep controlled mode
   tokenizer-agnostic. For long documents the difference vs. BPE
   tokenizers is small, but not zero.
3. **Hyperlink-based grouping requires pre-populated metadata.** The
   canonical dataset suite does not currently supply
   `metadata["hyperlinks"]` for Wikipedia-style corpora, so the adapter
   falls back to `one_document_per_unit` with a warning when this
   policy is requested. This is a dataset-side concern, not an
   adapter-side one.
4. **Native mode is untested on GPU in CI.** The smoke test is
   `skipif`-guarded and is designed to run on the university GPU
   machine, not in the local pytest run.
5. **No FAISS backend.** The adapter uses an in-memory numpy index
   persisted under `work_dir`. This is not architecture-defining for
   LongRAG (the official repo uses FAISS, but index backend is a
   storage choice, not an algorithm). The retrieval logic - long units,
   top-k1 units, MaxP expansion - is preserved exactly.

---

## 11. Deviations from the contract

None. The adapter conforms to `docs/adapter_contract.md` v2:

- subclasses `BaseBaseline`;
- overrides the identity attributes;
- matches the constructor signature exactly;
- implements `build_index()` and `retrieve()`;
- persists artifacts under `self.work_dir`;
- returns contiguous `RetrievedChunk` ranks;
- both `config_native.yaml` and `config_controlled.yaml` conform to
  the frozen schema;
- no imports from `rag_system`;
- no hardcoded absolute paths;
- no network calls in tests.

---

## 12. Graduation request

The adapter is **ready for graduation** from `workspace/longrag_dev/`
to `baselines/longrag/` after the supervisor:

1. Confirms `python -m pytest baselines.longrag_v2\tests\ -v` passes
   (expected: 6 passed, 1 skipped);
2. Reviews `adapter.py` for contract compliance;
3. Moves the folder to `baselines/longrag/`;
4. Updates `configs/baselines_registry.yaml` with `status: executable`.

This adapter does **not** graduate itself, per the operating manual
section 10.