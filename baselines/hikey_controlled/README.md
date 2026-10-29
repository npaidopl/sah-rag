# HiKEY Controlled Baseline (SAH-RAG v2)

**Baseline ID:** `hikey_controlled`
**Adapter version:** 0.1.0
**Status:** EXECUTABLE (controlled track only)
**Modes:** controlled = yes | native = NO (reported-only)

---

## 1. Reference

- **Paper:** Joongmin Shin, Gyuho Shim, Jeongbae Park, Jaehyung Seo,
  Heuiseok Lim. *"HiKEY: Hierarchical Multimodal Retrieval for
  Open-Domain Document Question Answering."* ACL 2026 (Main).
- **arXiv:** <https://arxiv.org/pdf/2605.29606>
- **ACL Anthology:** `2026.acl-long.818`
- **Project page:** <https://gyuhoshim.github.io/projects/hikey/>

---

## 2. Disposition (why this adapter looks the way it does)

| Track | Status | Reason |
|---|---|---|
| **Native (paper-exact / author-supported)** | ❌ Not supported | No public repository, no author-released configs, no model weights. Native execution is impossible without inventing a system that is not HiKEY. |
| **Native (reported-only)** | ✅ Supported | Paper-reported metrics are exposed as `HiKEYBaseline.REPORTED_METRICS` for downstream comparison tables. |
| **Controlled (paper-derived)** | ✅ Supported | Text-only approximation that preserves architecture-defining logic: coarse-to-fine routing, MaxSim section scoring, governing-header evidence assembly. |

**Frozen rule (`not-HiKEY`):** This adapter is a *paper-derived text-only approximation*. It is **not** HiKEY. It does **not** participate in the Native Track. Every `RetrievedChunk` carries `metadata["approximation"] = "hiKEY_text_only_approximation"` as a hard marker.

### What is preserved

- **Coarse-to-fine retrieval**: Stage 1 routes to top-N documents (hybrid BM25 + dense), Stage 2 scores sections within those documents.
- **MaxSim section scoring**: A section's score is `max` over its intra-section unit similarities, not `mean` or `sum`. This is HiKEY's defining aggregation choice.
- **Governing-header evidence assembly**: Retrieved section text is prefixed with `[parent_title] [own_title]` to preserve ancestry context.
- **Hierarchy-aware parsing**: Prefers `metadata["sections"]` (with `title`, `text`, `level`, `parent_id`) when present; falls back to deterministic size-based sectioning otherwise.
### What is NOT preserved (out of scope for v2)

- **PDF layout parsing (DHP)** — v2 is IR-only and receives `CanonicalDocument` records. Layout parsing belongs to the dataset-adapter stage.
- **Tables, figures, captions as first-class units** — v2 canonical schema is text-based. Multimodal input is not available in this phase.
- **Modality-specific encoders** — only BGE-M3 (injected) is used in controlled mode.
- **LVLM reader / answer generation** — v2 has no generation stage at all (`docs/adapter_contract.md` §9).

---

## 3. Modes

### 3.1 Controlled mode (fully supported)

The orchestrator injects:
- `embedder` = BGE-M3 (shared with SAH-RAG)
- `llm` = shared model (unused here — `is_llm_based = False`)

The adapter builds its own index structure (hierarchy + BM25 stats + dense embeddings) under `self.work_dir` and uses the injected embedder for all encoding. It never calls external APIs.

### 3.2 Native mode (rejected)

`HiKEYBaseline.__init__` raises `ValueError` when `mode=RunMode.NATIVE`, because `supports_native=False`. This is intentional. Paper-reported metrics live in the class attribute:

    HiKEYBaseline.REPORTED_METRICS = {
        # "dataset_id": {"metric_name": value, ...},
    }

This dict will be populated from the paper when the native comparison table is assembled. Until then it stays empty — an empty dict is a truthful statement that *no code-verified native execution exists*.

---

## 4. Files

| File | Purpose |
|---|---|
| `__init__.py` | Exports `HiKEYBaseline` |
| `adapter.py` | The adapter implementation |
| `config_native.yaml` | Conformance only — documents the reported-only decision |
| `config_controlled.yaml` | Real config used by the orchestrator |
| `requirements.txt` | Empty (no extra pip dependencies) |
| `tests/test_adapter.py` | Offline tests (<3s on CPU) |
| `README.md` | This file |

---

## 5. Config schema (controlled)

    name: hikey
    mode: controlled
    top_k: 10
    extra:
      doc_route_top_n: 30     # Stage 1: keep top-N documents
      section_size: 2000      # fallback section size (chars)
      unit_size: 500          # unit size for MaxSim within a section
      dense_weight: 0.7       # hybrid score weight (dense)
      lexical_weight: 0.3     # hybrid score weight (BM25)

All keys except `name/mode/top_k/extra` are violations of the contract. Extra knobs go inside `extra`.

---

## 6. Testing

    python -m pytest workspace\hikey_dev\tests\ -v

Expected: **6 passed in <5s on CPU**.

Tests are offline. `MockEmbedder` is a deterministic SHA-256-based embedder, so no network and no model download is required in CI.

---

## 7. Graduation checklist

- [x] Subclass of `BaseBaseline`
- [x] Identity attributes set
- [x] `__init__` signature matches the frozen contract
- [x] `build_index()` persists under `self.work_dir`
- [x] `retrieve()` returns contiguous rank 1..k `RetrievedChunk`
- [x] `config_native.yaml` and `config_controlled.yaml` present
- [x] `tests/test_adapter.py` passes offline in <10s
- [x] `README.md` documents description, paper, license, status, deps
- [x] `requirements.txt` present (empty — none needed)
- [x] No imports from `rag_system`
- [x] No hardcoded absolute paths
- [x] No network calls in tests
- [x] No modifications outside `workspace/hikey_dev/`

**Status: ready to graduate to `baselines/hikey/`.**

---

## 8. Known limitations

1. **Native track is reported-only** — there is no public HiKEY code. Any claim of native HiKEY execution in this project must be refused. The only legitimate native comparison is *paper-reported numbers vs. our reproduced numbers*, clearly labelled.
2. **Text-only approximation** — tables, figures, and the LVLM reader are absent. Reports using this adapter must state so explicitly. Do not claim modality-level HiKEY equivalence.
3. **`REPORTED_METRICS` is empty** — it must be filled from the paper before any native comparison table is produced. This is a deliberate, visible gap, not an omission.
4. **`doc_id` semantics** — `RetrievedChunk.doc_id` is the parent document ID (as required by `docs/adapter_contract.md` §3). `chunk_id` is the section ID. Recall@k on `doc_id` therefore measures document-level retrieval, which matches how HiKEY routes at Stage 1.

---

## 9. Deviation from contract

**None.** The adapter implements only `build_index()` and `retrieve()`. No extra public methods, no private state leaked outside `self.*`, no `generate()`, no external calls.

---

## 10. Citation

BibTeX entry:

    @inproceedings{shin2026hikey,
      title     = {HiKEY: Hierarchical Multimodal Retrieval for Open-Domain
                   Document Question Answering},
      author    = {Shin, Joongmin and Shim, Gyuho and Park, Jeongbae and
                   Seo, Jaehyung and Lim, Heuiseok},
      booktitle = {Proceedings of the 64th Annual Meeting of the Association
                   for Computational Linguistics (ACL 2026)},
      year      = {2026},
      note      = {arXiv:2605.29606}
    }
