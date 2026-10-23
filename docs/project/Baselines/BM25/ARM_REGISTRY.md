# BM25 Arm Registry & Handoff Notes

> Handoff document from the baseline developer to the SAH-RAG supervisor.
> Purpose: declare the three arm IDs the BM25 baseline occupies, explain why
> `NATIVE β‰΅ CONTROLLED` for BM25 (to prevent double-counting), and enumerate
> what remains before graduation to `baselines/bm25/`.

---

## 1. Arm IDs

Per the SAH-RAG v2 project plan, every baseline may appear in up to three
"arms" corresponding to the three research stages:

| Arm ID | Stage | Role | Config | Runs? |
|---|---|---|---|---|
| `BM25_NATIVE` | (Ξ±) native end-to-end | reference: BM25 as the author designed it | `config_native.yaml` | 1Γ— |
| `BM25_CONTROLLED` | (Ξ²) controlled adaptation | reference: BM25 inside the common post-retrieval stack | `config_controlled.yaml` | **same run as NATIVE** (see Β§2) |
| `BM25_REFERENCE` | (Ξ³) ablation | lower bound for SAH-RAG's gain | same run | **same run as NATIVE** (see Β§3) |

---

## 2. Why `NATIVE β‰΅ CONTROLLED` for BM25

The `BaseBaseline` contract defines `controlled` mode as:

> the orchestrator injects `embedder` (BGE-M3) and `llm` (shared model). The
> baseline MUST use the injected components for embedding and generation.
> β€” `docs/adapter_contract.md` Β§4

BM25 is **architecturally incompatible** with that injection:

1. **No dense embedder.** BM25 is lexical (bag-of-words). It cannot "use"
   BGE-M3 for embedding without becoming a fundamentally different retriever
   (hybrid sparse+dense), which is not BM25 anymore.
2. **No generation.** `is_llm_based = False`. BM25 emits ranked chunks; it
   never calls an LLM.

**Consequence:** in `controlled` mode the adapter accepts `embedder` and `llm`
(required by the frozen `BaseBaseline.__init__` signature) but **ignores both**.
The Lucene index, BM25 scoring (`k1=0.9, b=0.4`), tokenization, and result
ranking are identical to `native` mode.

Therefore:

- **`BM25_NATIVE` and `BM25_CONTROLLED` produce byte-identical retrieval
  results on the same corpus.**
- We run BM25 **once**. The same run file serves both arms.
- This is **not** a contract violation. It is the expected behavior for a
  baseline whose algorithm has no shareable dense or LLM components. It must
  be declared in any report or paper that tabulates BM25 alongside SAH-RAG.

### How this is declared in reports

> *"BM25 is a lexical baseline with no dense embedder and no LLM generation.
> In controlled mode, injected components were accepted per the SAH-RAG
> adapter contract but structurally unused, because BM25's algorithm is
> bag-of-words. Consequently `BM25_NATIVE` and `BM25_CONTROLLED` results are
> identical by construction. The same run file is reported for both arms."*

---

## 3. How BM25 feeds SAH-RAG's ablation (stage Ξ³)

In the ablation study, BM25 is **not** an arm of SAH-RAG. It is an **external
lower bound**, shown as a reference row alongside:

- `SAH-RAG_FULL`
- `SAH-RAG_ β’ component` (one row per ablated component)
- `SAH-RAG_ β’ BM25` β€” **only if** SAH-RAG internally uses BM25 / sparse
  retrieval. If it doesn't, this row does not exist.

The `BM25_REFERENCE` label is therefore a **view** of the `BM25_NATIVE` run
restricted to the (Ξ³) reporting table. No new run is executed.

If SAH-RAG turns out to contain BM25 internally, `SAH-RAG_ β’ BM25` becomes a
genuine ablation arm and needs its own run β€” but that is the SAH-RAG team's
responsibility, not this baseline's.

---

## 4. What is (and isn't) the BM25 adapter's responsibility

| In scope (this baseline) | Out of scope (elsewhere) |
|---|---|
| Executable BM25 adapter under `baselines/bm25/` | Dataset selection |
| Native + controlled mode support (with Β§2 caveat) | Generation evaluation (v2 is IR-only) |
| Pass the offline test suite | Modifying `configs/baselines_registry.yaml` |
| Document license, deps, known limitations | Graduating the folder to `baselines/bm25/` |
| Provide a reproducible `build_index β†’ retrieve` flow | Comparing numbers across baselines |

The adapter's job ends when the supervisor approves graduation. The BM25
adapter is **`TOTALLY_READY`** as of the last test run.

---

## 5. Graduation checklist (for the supervisor)

Everything below is verifiable by re-running the commands in the README.

- [x] All 7 files present (`__init__.py`, `adapter.py`, `config_native.yaml`,
      `config_controlled.yaml`, `README.md`, `requirements.txt`,
      `tests/test_adapter.py`)
- [x] `adapter.py` subclasses `BaseBaseline` with the frozen `__init__`
      signature
- [x] `retrieve()` returns contiguous `RetrievedChunk` ranks 1..k
- [x] `build_index()` persists under `self.work_dir` and is idempotent
- [x] No hardcoded absolute paths
- [x] No external API calls
- [x] No imports from `rag_system`
- [x] Both `config_native.yaml` and `config_controlled.yaml` present with the
      contract-mandated schema (`name`, `mode`, `top_k`, `extra`)
- [x] `README.md` documents description, paper, license, status, deps, and
      known limitations
- [x] `tests/test_adapter.py` passes offline in ~16-20s (JVM startup dominates;
      see README "Timing note")
- [x] pytest faulthandler false-positive noise silenced via
      `tests/conftest.py` (monkeypatch, not config change)
- [x] No modifications outside `baselines/bm25/`

**Suggested next steps (supervisor):**

1. Run `python -m pytest baselines/bm25\tests\ -v` once to confirm.
2. Review `adapter.py` for the structural exception in Β§2.
3. Move `baselines/bm25` β†’ `baselines/bm25`.
4. Add an entry to `configs/baselines_registry.yaml`:

   ```yaml
   - name: bm25
     status: executable
     supports_native: true
     supports_controlled: true
     is_llm_based: false
     notes: "Lexical baseline. In controlled mode, injected components are structurally unused; see baselines/bm25/ARM_REGISTRY.md Β§2."
   ```

5. Commit with message: `feat(baselines): add bm25 adapter (Pyserini/Anserini)`.

---

## 6. Consolidated known limitations

- **Java 21 required.** Pyserini/Anserini loads the JVM via `jnius`. Java
  versions other than 21 (e.g., 24) cause access violation crashes during
  JVM initialization on Windows. The adapter does not enforce this; the README
  does.
- **`k1` / `b` overrides are recorded but not applied.** The Pyserini CLI
  uses Lucene defaults. Custom `Similarity` is out of scope for v0.1.0.
- **Document-level only.** BM25 has no chunking; `chunk_id == doc_id`.
- **JVM startup cost.** Cold test run is ~16-20s. Real benchmark runs will
  amortize the JVM cost over the dataset.
- **Controlled-mode injection is structurally unused** (see Β§2).

---

## 7. Reproduction reference (placeholder)

Once a dataset is selected, the native-mode run must match the official
Pyserini regression numbers for that dataset. Common reference points:

| Dataset | k1 | b | Reported metric | Reference |
|---|---|---|---|---|
| MS MARCO passage (v1) | 0.9 | 0.4 | MRR@10 = 0.187 | Pyserini README |
| MS MARCO passage (v2) | 0.9 | 0.4 | nDCG@10 β‰ 0.228 | TREC DL 2019 |
| BEIR (per dataset) | 0.9 | 0.4 | nDCG@10 varies | BEIR paper |

Discrepancies imply version drift in `pyserini`, Anserini, or Lucene and must
be resolved by pinning versions. Pin in effect: `pyserini==2.4.0`.

---

## 8. Changelog

- **v0.1.0** β€” initial adapter, native + controlled mode, offline test with
  faulthandler monkeypatch, README, ARM_REGISTRY. Status: `TOTALLY_READY`
  (workspace-only, pending supervisor graduation).


---

## 9. Verified native-fidelity tier (SciFact)

**Date:** 2026-10-22

The BM25 adapter was verified end-to-end against the official Pyserini
SciFact pipeline. Result:

| Dataset | Tier | Evidence |
|---|---|---|
| BEIR SciFact | **PAPER_EXACT_NATIVE** | nDCG@10 identical (0.678909); top-10 docid overlap 300/300 queries; Docids AND scores are identical per query. Adapter's independently-built index reproduces the official retrieval output exactly. |

### What this means for (Ξ±) native E2E

For SciFact, the adapter can be reported at **PAPER_EXACT_NATIVE** fidelity. When
the SAH-RAG dataset is selected, the same tier framework will be re-applied
and documented in this section.

### Why not byte-identical

The adapter and official CLI produce **identical retrieval output** but
different run file bytes, due to:
- Different --hits (1000 vs 10) β€” adapter respects the contract's 	op_k
- Different tag field (Anserini vs m25_adapter)
- Different qid iteration order (Pyserini CLI vs Python dict)

Byte-identical was never the target; behavioral equivalence was. Behavioral
equivalence is confirmed at nDCG@10 = 0.678909 and 300/300 top-10 docid match.


---

## 10. Smoke-test verification (end-to-end, real data)

**Date:** 2026-10-23 14:23:19

Two smoke tests were run with **real data from disk** (not synthetic) to
prove the full pipeline works end-to-end, per the "one-button-press"
requirement for the eventual SAH-RAG dataset run.

### 10.1 Native E2E smoke

- Script: `baselines/bm25/scripts/smoke_native_e2e.py`
- Data: 30 real SciFact docs, 5 real SciFact queries (from disk)
- Result: **PASSED** — CanonicalDocument -> build_index -> retrieve -> ranked output validated
- Output structure validated: contiguous ranks, all doc_ids in corpus, chunk_id == doc_id
- Runtime after JVM init: ~4s (build + 5 queries)

### 10.2 Controlled-adaptation smoke (with real BGE-M3)

- Script: `baselines/bm25/scripts/smoke_controlled_e2e.py`
- Data: same 30 docs, 5 queries
- Embedder: **real BAAI/bge-m3** (4.27 GB, local snapshot) — not a mock
- Result: **PASSED**
  - BGE-M3 loaded and produced 1024-dim dense vectors
  - Native == Controlled output: identical docids, ranks, scores
  - Embedder encode calls by BM25: **0** (structural exception confirmed)
- Protocol integrity: shared embedder injected via contract, structurally unused

### 10.3 Performance benchmark (steady state)

- Script: `baselines/bm25/scripts/bench_retrieval.py`
- 500 docs, 50 queries
- JVM init (first query): 12.58s
- Warm per-query (median): **3.3 ms**
- Steady-state: **301.7 q/s**
- MS MARCO projection: ~0.4 min for 6,980 queries

### 10.4 Implication

The adapter is ready for the SAH-RAG dataset. When it arrives, only the
dataset path needs to change in the smoke scripts. No adapter code change
is expected. The "one-button-press" requirement is satisfied: the scripts
are standalone, use venv python directly, and validate output structure.
