# BM25 Baseline — Final Report

**Date:** 2026-10-23 00:04:30
**Location:** `baselines/bm25/` (graduated from `workspace/bm25_dev/`)
**Status:** COMPLETE — graduated to production baselines.

---

## 1. Executive summary

The BM25 baseline for the SAH-RAG v2 project is **functionally complete,
verified, and graduated**. It uses the official Pyserini/Anserini
implementation (Lucene BM25Similarity, `k1=0.9, b=0.4`) and reproduces
the canonical baselines at both small and large scale.

| Check | Result |
|---|---|
| SciFact native reproduction | nDCG@10 = 0.678909 (official flat = 0.6789) |
| SciFact adapter E2E | nDCG@10 = 0.678909 (identical) |
| SciFact native-fidelity | **PAPER_EXACT_NATIVE** (300/300 queries, docid+score identical) |
| MS MARCO passage v1 | MRR@10 = 0.187486 (published = 0.187) |
| Controlled vs native | Structurally identical (automated test) |
| Contract compliance | 3/3 tests pass from new location |

---

## 2. Deliverables

### (α) Native end-to-end — VERIFIED

| Dataset | Index | Metric | Ours | Reference | Δ |
|---|---|---|---|---|---|
| BEIR SciFact | `beir-v1.0.0-scifact.flat` | nDCG@10 | 0.678909 | 0.6789 | 9e-06 |
| MS MARCO passage v1 | `msmarco-passage` | MRR@10 | 0.187486 | 0.187 | 0.00049 |

**Native-fidelity classification: PAPER_EXACT_NATIVE for SciFact**

The adapter builds its own Lucene index from `CanonicalDocument` objects
and produces **byte-equivalent retrieval behavior** to the official Pyserini
pipeline: 300/300 queries match in top-10 docids AND scores (max score delta
= 0.000000).

### (β) Controlled-adaptation — VERIFIED

For BM25, native and controlled mode produce **identical** retrieval output,
because BM25 is:
- Lexical (no dense embedder to inject)
- Non-LLM-based (`is_llm_based = False`; no generation)

The adapter accepts injected `embedder` and `llm` per the frozen
`BaseBaseline` contract but **structurally ignores both**. This is proven
by `tests/test_controlled_mode.py::test_controlled_equivalent_to_native`.

This is **not** a contract violation — it reflects BM25's architecture and
is documented in `ARM_REGISTRY.md` section 2.

### (γ) Ablation reference — OPTIONAL, pending request

The `BM25_REFERENCE` arm will be a **view** of the native run when the
SAH-RAG team requests it. No new run will be executed. If SAH-RAG contains
an internal sparse/BM25 component, `SAH-RAG_ − BM25` becomes a genuine
ablation arm — but that is the SAH-RAG team's responsibility, not this
baseline's.

---

## 3. Evidence artifacts

| Artifact | Location | Value |
|---|---|---|
| SciFact official run | `runs/scifact.bm25.trec` | nDCG@10 = 0.678909 (294K lines) |
| SciFact adapter run | `runs/scifact.adapter_e2e.trec` | nDCG@10 = 0.678909 (3K lines) |
| MS MARCO official run | `runs/msmarco_passage_v1.bm25.trec` | MRR@10 = 0.187486 (6.97M lines) |
| Stage 1 report | `reports/stage1_scifact.md` | Reproduction check (PASS) |
| Stage 2 report | `reports/stage2_msmarco.md` | Credibility check (PASS) |
| Adapter E2E report | `reports/adapter_e2e_scifact.md` | PAPER_EXACT_NATIVE confirmation |
| E2E log | `reports/adapter_e2e_scifact.log` | Full execution log |

Note: `runs/*.trec` and `*.log` are excluded from git via `.gitignore`
(they are regenerable evidence, not source). They remain on disk.

---

## 4. Files in baselines/bm25/

| File | Purpose |
|---|---|
| `adapter.py` | `BM25Baseline` (subclasses frozen `BaseBaseline`) |
| `config_native.yaml` | Native config (k1=0.9, b=0.4) |
| `config_controlled.yaml` | Controlled config |
| `README.md` | Description, license, deps, limitations |
| `ARM_REGISTRY.md` | Arm IDs, native-fidelity tier, scope |
| `DATASET_SELECTION.md` | Dataset decision record |
| `GRADUATION_PACKAGE.md` | Handoff doc (used for this graduation) |
| `FINAL_REPORT.md` | This file |
| `requirements.txt` | `pyserini==2.4.0` (Java 21 system requirement) |
| `.gitignore` | Excludes caches + regenerable artifacts |
| `scripts/eval_ndcg.py` | Windows-safe pytrec_eval wrapper |
| `scripts/run_adapter_e2e_scifact.py` | Adapter E2E reproduction script |
| `tests/conftest.py` | Faulthandler monkeypatch (Windows+JVM) |
| `tests/test_adapter.py` | Base build+retrieve test |
| `tests/test_controlled_mode.py` | Native ≡ Controlled verification |

---

## 5. Environment

- Python 3.12
- Java 21 (Temurin 21.0.12.1) — **required**; Java 22+ causes JVM/jnius crashes
- `pyserini==2.4.0`
- `pytrec_eval_terrier==0.5.11`
- Cache: `D:\rag_work\pyserini_cache` (2.5 GB MS MARCO, 10 MB SciFact)

---

## 6. Known limitations

1. **Java 21 required.** `pyserini==2.4.0` + `jnius` cannot load JVM on
   Java 22+ without `access violation` faults. Documented in README.
2. **`k1`/`b` overrides recorded but not applied.** Pyserini CLI uses
   Lucene defaults. Custom `Similarity` requires a Pyserini fork (out of
   scope for v0.1.0).
3. **Document-level only.** BM25 has no chunking; `chunk_id == doc_id`.
4. **Controlled-mode injection structurally unused** (see ARM_REGISTRY §2).
5. **pytest faulthandler monkeypatch.** Windows+JVM interaction;
   silenced at `tests/conftest.py` level without touching project config.
6. **Cold JVM startup ~10-15s.** Amortized in real workloads (19s test run).

---

## 7. Verification

    .\.venv\Scripts\Activate.ps1
    python -m pytest baselines\bm25\tests\ -v

Expected: 3 tests pass (skip only if Java 21 not on PATH).

---

## 8. Graduation trace

| Step | Result |
|---|---|
| Pre-checks | source + dest + registry OK |
| Registry backup | `baselines_registry.yaml.bak.20261022-230853` |
| Copy | 21 files → `baselines/bm25/` |
| Imports updated (.py) | 3 files (`workspace.bm25_dev` → `baselines.bm25`) |
| Refs updated (.md) | 7 files (`workspace/bm25_dev` → `baselines/bm25`) |
| `.gitignore` | created |
| Tests from new location | 3 passed in 24.70s |
| `workspace/bm25_dev` | removed |
| Registry | appended `bm25` entry (APPENDED) |

---

## 9. Supervisor next steps

1. Run tests one more time to confirm:
   
       python -m pytest baselines\bm25\tests\ -v

2. Stage and commit:
   
       git add baselines\bm25\ configs\baselines_registry.yaml
       git commit -m "feat(baselines): add bm25 adapter (Pyserini/Anserini)"

3. After verification, delete the registry backup:
   
       Remove-Item configs\baselines_registry.yaml.bak.*

---

## 10. After graduation — orchestrator integration

Once committed, the orchestrator can invoke the BM25 baseline via
`BaseBaseline` for both modes:

- `BM25_NATIVE` → `baselines/bm25/config_native.yaml`
- `BM25_CONTROLLED` → `baselines/bm25/config_controlled.yaml`
  (identical output to native; single run suffices)

Results flow into SAH-RAG comparison tables per `ARM_REGISTRY.md §1-2`.
The `BM25_REFERENCE` (γ) arm is a view of the same run — no extra execution.

When the SAH-RAG dataset is selected, re-verify the (α) native E2E on it
using `scripts/run_adapter_e2e_scifact.py` as template. No adapter code
change expected.

---

## 11. Handoff signature

- Adapter version: 0.1.0
- Contract version: v2 (frozen)
- Native-fidelity tier: PAPER_EXACT_NATIVE (SciFact)
- Graduated: 2026-10-23 00:04:30
- Graduated by: supervisor-authorized move
- All work contained under `baselines/bm25/` post-graduation