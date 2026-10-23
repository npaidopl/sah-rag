# Αναφορά Κατάστασης — BM25 Baseline (v3.0, Final)

**Ημερομηνία:** 2026-10-23
**Για:** SAH-RAG Organizer
**Status:** ✅ ΟΛΟΚΛΗΡΩΘΗΚΕ — και τα 3 deliverables (α/β/γ) ικανοποιημένα

> Η παρούσα αναφορά αντικαθιστά τις v0.1 (planning) και v2.0 (graduation).
> Περιλαμβάνει τα αποτελέσματα των smoke tests με πραγματικά δεδομένα.

---

## 1. Executive Summary

Η BM25 baseline είναι **πλήρως έτοιμη** για τα τελικά datasets της SAH-RAG:

| Deliverable | Status | Evidence |
|---|---|---|
| (α) Native E2E | ✅ COMPLETE | SciFact nDCG@10 = 0.678909 (PAPER_EXACT_NATIVE), MS MARCO MRR@10 = 0.187486, 30-doc smoke passed |
| (β) Controlled-adaptation | ✅ COMPLETE | Real BGE-M3 loaded + injected + 0 encode calls by BM25, native ≡ controlled verified |
| (γ) Ablation idea | ✅ DELIVERED | `baselines/bm25/ABLATION_IDEA.md` — 4 concrete proposals for SAH-RAG team to evaluate |

**«Πάτημα ενός κουμπιού» ready:** Όταν έρθει το SAH-RAG dataset, μόνο το
dataset path αλλάζει. Καμία αλλαγή κώδικα.

---

## 2. Τα 3 Deliverables — Αναλυτικά

### (α) Native E2E — COMPLETE

| Check | Result |
|---|---|
| SciFact full reproduction (official CLI) | nDCG@10 = 0.678909 vs 0.6789 → PASS |
| SciFact adapter E2E (own index) | nDCG@10 = 0.678909 → **PAPER_EXACT_NATIVE** |
| Native-fidelity verification | 300/300 queries docid+score identical (max delta 0.000000) |
| MS MARCO passage v1 | MRR@10 = 0.187486 vs 0.187 → PASS |
| 30-doc real-data smoke | PASSED — CanonicalDocument → index → retrieve → valid output |
| Output validation | Contiguous ranks, all doc_ids in corpus, chunk_id == doc_id |

### (β) Controlled-adaptation — COMPLETE

**Protocol** (per original spec): same corpus, same queries, shared embedder
(BGE-M3), shared LLM, same evaluation. For BM25, this means: BGE-M3 injected
via frozen contract, structurally unused (BM25 is lexical).

**Verified:**

| Check | Result |
|---|---|
| Real BGE-M3 loaded | ✅ 4.27 GB model, 1024-dim dense output |
| Contract validation passes | ✅ BM25 accepts embedder param |
| Native ≡ Controlled | ✅ Identical docids, ranks, scores for all queries |
| BM25 calls embedder.encode_* | **0 times** — structural exception confirmed |
| Smoke test (30 docs, 5 queries) | PASSED |

**Methodological note** (ARM_REGISTRY §2): BM25 native ≡ controlled is not
a bug or contract violation — it's the expected behavior for a lexical
baseline with no dense/LLM components. This must be declared in controlled-
mode comparison tables.

### (γ) Ablation idea — DELIVERED

Παρέδωσα το `baselines/bm25/ABLATION_IDEA.md` με 4 συγκεκριμένες προτάσεις
για την SAH-RAG ομάδα:

1. **Idea A — BM25 as first-stage candidate retriever** (medium effort, medium gain)
2. **Idea B — BM25 score as re-ranking feature** (low-medium effort, best ROI)
3. **Idea C — BM25+RM3 query expansion pre-processing** (low effort, small gain)
4. **Idea D — BM25 as routing signal** (speculative)

**Σύσταση:** Idea B (BM25 score as re-ranking feature) είναι πιο tractable.
Η SAH-RAG ομάδα αποφασίζει με το κύριο LLM. Εγώ δεν δεσμεύομαι να υλοποιήσω
καμία από αυτές — είναι ideas για αξιολόγηση.

---

## 3. Τοπικά Paths

**Project root:** `C:\Users\USER1\Documents\SAH-RAG\`

**Baseline:**
- `baselines/bm25/` — κώδικας, configs, tests, reports
- `baselines/bm25/FINAL_REPORT.md` — comprehensive report
- `baselines/bm25/ARM_REGISTRY.md` — arm IDs + native-fidelity tier (updated §10)
- `baselines/bm25/ABLATION_IDEA.md` — 4 ablation proposals (νέο)
- `baselines/bm25/scripts/` — 4 smoke/benchmark scripts
- `baselines/bm25/tests/` — 3 tests (all pass)

**Cache (D:, gitignored):**
- `D:\rag_work\pyserini_cache\` — 2.5 GB (SciFact + MS MARCO indexes)
- `D:\rag_work\hf_cache\models--BAAI--bge-m3\` — 4.27 GB
- `D:\rag_work\_bm25_smoke_native_idx\` — smoke test index
- `D:\rag_work\_bm25_smoke_ctrl\` — controlled smoke index
- `D:\rag_work\_bm25_bench_idx\` — benchmark index

**Git:**
- Branch: `restructure/v2`
- Commits: `62a0a0d` (graduation), `dbf5c90` (perf fix)
- Pending: commit αυτού του cell (ARM_REGISTRY + ABLATION_IDEA)

---

## 4. Τι ΜΕΝΕΙ (supervisor actions)

**Άμεσα:**
1. `git add baselines/bm25/ARM_REGISTRY.md baselines/bm25/ABLATION_IDEA.md baselines/bm25/scripts/smoke_native_e2e.py baselines/bm25/scripts/smoke_controlled_e2e.py`
2. `git commit -m "docs(bm25): add smoke verification + ablation ideas"`

**Όταν η SAH-RAG ομάδα επιλέξει dataset:**
1. Αλλάζουμε το `SCIFACT_ROOT` στο `scripts/smoke_native_e2e.py` → νέο dataset path
2. Αλλάζουμε το `SCIFACT_ROOT` στο `scripts/smoke_controlled_e2e.py` → ίδιο
3. Τρέχουμε και τα δύο → αναμένεται PASS (καμία αλλαγή κώδικα)
4. Αν η SAH-RAG ομάδα θέλει full E2E metrics: προσαρμόζουμε το `scripts/run_adapter_e2e_scifact.py` για το νέο dataset

**Όταν η SAH-RAG ομάδα ζητήσει ablation:**
1. Επιλέγει μία από τις 4 Ideas (A/B/C/D)
2. Εγώ υλοποιώ μόνο ό,τι χρειάζεται (π.χ. RM3 config για Idea C — ~1 ώρα)
3. Δεν υπάρχει default commitment για καμία idea

---

## 5. Non-scope (ξεκάθαρο)

Τα παρακάτω ΔΕΝ ανήκουν στη BM25 baseline:
- `sah_rag/` (η κύρια αρχιτεκτονική) — δεν αγγίχτηκε
- `evaluation/`, `orchestration/`, `datasets/` — δεν αγγίχτηκαν
- `docs/adapter_contract.md`, `baselines/base.py` (frozen) — δεν αγγίχτηκαν
- `pyproject.toml` — δεν αγγίχτηκε
- Test/benchmark scripts — δεν αγγίχτηκαν
- `configs/baselines_registry.yaml` — μόνο authorized BM25 entry

---

## 6. Performance benchmark

| Metric | Value |
|---|---|
| Index build (500 docs) | 2.78s |
| JVM init (first query) | 12.58s (cold, unavoidable) |
| Warm per-query | 3.3 ms |
| Steady-state | **301.7 q/s** |
| MS MARCO projection | ~0.4 min for 6,980 queries |

**Το caching fix (v0.1.1)** εξάλειψε per-query searcher open/close overhead.
Commit: `dbf5c90`.

---

## 7. Reproduction commands

**Tests:**
C:\Users\USER1\Documents\SAH-RAG.venv\Scripts\python.exe -m pytest baselines\bm25\tests\ -v

Expected: 3/3 pass.

**Native smoke:**
C:\Users\USER1\Documents\SAH-RAG.venv\Scripts\python.exe baselines\bm25\scripts\smoke_native_e2e.py

Expected: SMOKE TEST PASSED.

**Controlled smoke (real BGE-M3):**

Expected: SMOKE TEST PASSED.

**Controlled smoke (real BGE-M3):**
C:\Users\USER1\Documents\SAH-RAG.venv\Scripts\python.exe baselines\bm25\scripts\smoke_controlled_e2e.py

Expected: CONTROLLED-ADAPTATION SMOKE PASSED.

---

## 8. Handoff Signature

- Adapter version: 0.1.1
- Contract version: v2 (frozen)
- Native-fidelity: PAPER_EXACT_NATIVE (SciFact)
- Graduation commit: 62a0a0d
- Perf commit: dbf5c90
- Total files in baselines/bm25/: 15+ (excluding runs/trec)
- Evidence: 3 native runs, 1 adapter run (gitignored, on disk)

**Recommendation:** BM25 baseline is production-ready. Awaiting SAH-RAG
dataset for final re-verification. All three deliverables satisfied.

**Next action:** Commit ARM_REGISTRY + ABLATION_IDEA updates, then wait
for SAH-RAG dataset selection.

---

**Υπογραφή:** BM25 Baseline Developer
**Ημερομηνία:** 2026-10-23