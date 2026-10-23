# Αναφορά Κατάστασης — BM25 Baseline (v2.0, Completion Update)

**Ημερομηνία:** 2026-10-23
**Για:** SAH-RAG Organizer
**Status:** ✅ ΟΛΟΚΛΗΡΩΘΗΚΕ ΚΑΙ ΠΑΡΑΔΟΘΗΚΕ

> Η παρούσα αναφορά **αντικαθιστά** την προηγούμενη v0.1 (planning stage).
> Όλα τα pending items της v0.1 είτε ολοκληρώθηκαν είτε εξαρτώνται πλέον
> αποκλειστικά από απόφαση της SAH-RAG ομάδας (dataset selection).

---

## 0. Executive Summary

Η BM25 baseline είναι **πλήρως έτοιμη, επαληθευμένη και graduated** στο
`baselines/bm25/`. Reproduces τις official Pyserini baselines τόσο σε small
scale (SciFact, 5K docs) όσο και σε large scale (MS MARCO, 8.8M docs), με
**PAPER_EXACT_NATIVE** classification για το SciFact.

| Deliverable | Status |
|---|---|
| (α) Native E2E | ✅ **PAPER_EXACT_NATIVE** (SciFact) + MRR@10 reproduction (MS MARCO) |
| (β) Controlled-adaptation | ✅ Verified (native ≡ controlled, automated test) |
| (γ) Ablation reference | ⏸️ Optional, pending SAH-RAG request |
| Graduation | ✅ Committed: `62a0a0d` on `restructure/v2` |

---

## 1. Τοπική Παρουσία & Πληρότητα

### α) Τα έχουμε τοπικά;

**✅ ΝΑΙ.** Ο κώδικας, οι tests, τα reports και τα evidence artifacts βρίσκονται όλα στο `baselines/bm25/`.

### β) Είναι πλήρη;

**✅ ΝΑΙ.** 3/3 tests περνούν από τη νέα θέση, registry entry ενεργό, git commit ολοκληρωμένο.

### γ) Ακριβή paths

**Project root:**

```text
C:\Users\USER1\Documents\SAH-RAG\
```

**Graduated baseline (canonical, committed):**

```text
baselines\bm25
├── adapter.py                      # BM25Baseline
├── config_native.yaml              # k1=0.9, b=0.4
├── config_controlled.yaml          # structural exception documented
├── README.md
├── ARM_REGISTRY.md                 # arm IDs + native-fidelity tier
├── DATASET_SELECTION.md            # decision record
├── GRADUATION_PACKAGE.md           # handoff doc
├── FINAL_REPORT.md                 # comprehensive report
├── requirements.txt                # pyserini==2.4.0
├── .gitignore                      # excludes runs/*.trec + *.log
├── __init__.py
├── reports
│   ├── stage1_scifact.md           # Stage 1 (PASS)
│   ├── stage2_msmarco.md           # Stage 2 (PASS)
│   ├── adapter_e2e_scifact.md      # PAPER_EXACT_NATIVE confirmation
│   └── adapter_e2e_scifact.log
├── scripts
│   ├── eval_ndcg.py                # Windows-safe pytrec_eval wrapper
│   └── run_adapter_e2e_scifact.py
├── tests
│   ├── conftest.py                 # faulthandler monkeypatch
│   ├── test_adapter.py
│   └── test_controlled_mode.py
└── runs\                           # gitignored, on disk as evidence
    ├── scifact.bm25.trec           # 10.9 MB (official)
    ├── scifact.adapter_e2e.trec    # 0.1 MB (adapter)
    └── msmarco_passage_v1.bm25.trec  # 274.2 MB (official)
```

**Registry:**

```text
configs\baselines_registry.yaml     # bm25 entry appended
```

**Cache (D:, not in repo):**

```text
D:\rag_work\pyserini_cache\         # 2.5 GB (MS MARCO + SciFact indexes)
D:\rag_work\hf_cache\               # HF_HOME (SciFact corpus)
D:\rag_work\torch_cache\            # placeholder (unused by BM25)
```

**Environment:**

```text
.venv\                                                      # Python 3.12
C:\Program Files\Eclipse Adoptium\jdk-21.0.12.101-hotspot\  # Java 21
```

**Removed (post-graduation):**

```text
workspace\bm25_dev\                                         # moved to baselines/bm25/ then deleted
configs\baselines_registry.yaml.bak.20261022-230853         # cleanup done
```

**Git:**

```text
Branch: restructure/v2
Commit: 62a0a0d feat(baselines): add bm25 adapter (Pyserini/Anserini)
Files:  20 changed, 1632 insertions(+)
```

### δ) Εργαλεία που χρησιμοποιήθηκαν (για reference)

| Tool | Version | Purpose |
|---|---|---|
| pyserini | 2.4.0 (pinned) | BM25 retrieval + indexer |
| pytrec_eval_terrier | 0.5.11 | Windows-safe trec_eval replacement |
| Java (Temurin) | 21.0.12.1 | Anserini/Lucene runtime |
| Lucene | 9.x (bundled) | Index backend |
| HF datasets | transitive | SciFact corpus download |
| pytest | 8.4.2 | Test runner |

---

## 2. Χρονολογική Σειρά Βημάτων

| # | Βήμα | Αποτέλεσμα |
|---|---|---|
| 1 | Methodology lockdown (arm registry, native fidelity 3-tier) | Defined BM25_NATIVE / BM25_CONTROLLED / BM25_REFERENCE |
| 2 | Adapter scaffolding (7 files) | `BM25Baseline` subclass του `BaseBaseline` |
| 3 | Test infrastructure (pytest + JVM fix) | Faulthandler monkeypatch, 1/1 pass |
| 4 | Cache redirect D: (2.5 GB off C:) | `PYSERINI_CACHE`, `HF_HOME`, `TORCH_HOME` set |
| 5 | Stage 1: SciFact reproduction | nDCG@10 = 0.678909 vs 0.6789 official → PASS |
| 6 | Stage 2: MS MARCO reproduction | MRR@10 = 0.187486 vs 0.187 → PASS (57.7 min) |
| 7 | Adapter-level SciFact E2E | nDCG@10 = 0.678909 (identical) |
| 8 | Native-fidelity verification | 300/300 queries, docid+score identical → **PAPER_EXACT_NATIVE** |
| 9 | (β) controlled-adaptation test | `test_controlled_equivalent_to_native` PASSED |
| 10 | Graduation package + move | `baselines/bm25/` (3/3 tests pass) |
| 11 | Registry update + git commit | `62a0a0d` |

### Τι έγινε στο τελευταίο βήμα (για συντομία)

**Graduation:** μετακίνηση `workspace/bm25_dev/` → `baselines/bm25/` με
rollback protection, ενημέρωση imports/refs, δημιουργία `.gitignore`,
re-run tests από τη νέα θέση (3/3 pass), διαγραφή workspace, προσθήκη
registry entry, git commit `62a0a0d`.

---

## 3. Status — (α) Native End-to-End

```text
STATUS: TOTALLY_READY
POSSIBILITY_TO_BE_DONE_SUCCESSFULLY: high
STUCK: No
```

**Πλήρης αναφορά:**

Reproduced official baselines σε δύο scales:

| Dataset | Index | Metric | Ours | Reference | Δ |
|---|---|---|---|---|---|
| BEIR SciFact | `beir-v1.0.0-scifact.flat` | nDCG@10 | 0.678909 | 0.6789 | 9e-06 |
| MS MARCO passage v1 | `msmarco-passage` | MRR@10 | 0.187486 | 0.187 | 0.00049 |

Επιπλέον, ο **adapter** (όχι το official CLI) παράγει **byte-equivalent
retrieval behavior**: 300/300 queries match in top-10 docids AND scores
(max delta = 0.000000). Classification: **PAPER_EXACT_NATIVE**.

**Εκκρεμή βήματα:**

- ⏸️ Re-verification στο **SAH-RAG dataset** όταν επιλεγεί. Δεν απαιτεί
  αλλαγή κώδικα — απλώς εκτέλεση του `scripts/run_adapter_e2e_scifact.py`
  προσαρμοσμένου στο νέο dataset (~30 min runtime). Εξαρτάται **αποκλειστικά**
  από την απόφαση της SAH-RAG ομάδας για dataset.

---

## 4. Status — (β) Controlled-Adaptation

```text
STATUS: TOTALLY_READY
POSSIBILITY_TO_BE_DONE_SUCCESSFULLY: high
STUCK: No
```

**Πλήρης αναφορά:**

Για τη BM25, το native και το controlled mode παράγουν **ταυτόσημη** έξοδο.
Αυτό προκύπτει δομικά (η BM25 δεν έχει dense embedder και δεν είναι LLM-based),
και **αποδεικνύεται εκτελεστικά**:

- `test_controlled_equivalent_to_native` → PASSED
- Χτίζει index σε native mode, μετά σε controlled mode (με dummy embedder),
  συγκρίνει top-10 docids + ranks + scores → **πανομοιότυπα**.

Το ότι δέχεται `embedder`/`llm` kwargs (frozen contract) αλλά τα αγνοεί
είναι **τεκμηριωμένη δομική εξαίρεση** (`ARM_REGISTRY.md §2`), **όχι**
παράβαση του contract.

**Εκκρεμή βήματα:**

- ⏸️ Καμία εκτέλεση δεν απαιτείται. Μία native run αρκεί για τα δύο arms.
  Re-verification (αν ζητηθεί) θα γίνει αυτόματα όταν τρέξει το (α) στο
  SAH-RAG dataset — τα αποτελέσματα θα είναι ταυτόσημα.

---

## 5. Status — (γ) Ablation Reference (Optional)

```text
STATUS: in process (deferred)
POSSIBILITY_TO_BE_DONE_SUCCESSFULLY: high
STUCK: Yes
Reason: Αναμένει αίτημα από την SAH-RAG ομάδα. Θα παραχθεί ως view του
        ίδιου native run — δεν απαιτεί νέα εκτέλεση. Το πότε θα
        χρειαστεί εξαρτάται από το design του ablation table της SAH-RAG.
Next_Step: Όταν η SAH-RAG ομάδα ολοκληρώσει το ablation design και ζητήσει
        το BM25 reference row, το παρέχουμε από τα ήδη υπάρχοντα runs
        (καμία νέα εκτέλεση, μηδενικό κόστος).
```

**Σημείωση:** Αν η SAH-RAG περιέχει εσωτερικό sparse/BM25 component, τότε
το `SAH-RAG_ − BM25` γίνεται **πραγματικό ablation arm** και θα απαιτήσει
νέο run — αλλά αυτό είναι responsibility της SAH-RAG ομάδας, όχι της BM25
baseline.

---

## 6. Σύγκριση με προηγούμενη αναφορά (v0.1 → v2.0)

| Item v0.1 | Status v2.0 |
|---|---|
| Adapter scaffolding | ✅ Ολοκληρώθηκε |
| BM25 reproduction (SciFact) | ✅ PASS (0.678909) |
| BM25 reproduction (MS MARCO) | ✅ PASS (0.187486) |
| (α) Native E2E | ✅ VERIFIED (PAPER_EXACT_NATIVE) |
| (β) Controlled-adaptation | ✅ VERIFIED (test passes) |
| (γ) Ablation | ⏸️ Deferred (αναμένει SAH-RAG request) |
| Graduation | ✅ Committed `62a0a0d` |
| Re-verification SAH-RAG dataset | ⏸️ Pending dataset selection |

---

## 7. Τι ΜΕΝΕΙ (Supervisor Actions)

**Άμεσα:**

1. ✅ `git add` + `git commit` — **DONE** (`62a0a0d`)
2. ✅ Backup cleanup — **DONE**

**Όταν η SAH-RAG ομάδα επιλέξει dataset:**

1. Re-run `scripts/run_adapter_e2e_scifact.py` προσαρμοσμένο στο νέο dataset
   (αναμενόμενα ~30 min, **καμία αλλαγή κώδικα**).
2. Επιβεβαίωση ότι το native-fidelity tier παραμένει PAPER_EXACT_NATIVE
   (αναμένεται).
3. Update `ARM_REGISTRY.md §9` με το νέο dataset row.

**Όταν η SAH-RAG ομάδα ζητήσει ablation table:**

1. Export view από τα existing native runs ως `BM25_REFERENCE` row.
2. Αν SAH-RAG έχει εσωτερικό BM25: run `SAH-RAG_ − BM25` (SAH-RAG team's job).

---

## 8. Non-scope (ξεκάθαρο)

Τα παρακάτω **δεν** ανήκουν στη BM25 baseline και δεν αγγίχτηκαν:

- `sah_rag/` (η κύρια αρχιτεκτονική)
- `evaluation/`, `orchestration/`, `datasets/`
- `docs/adapter_contract.md`, `baselines/base.py` (frozen)
- `pyproject.toml`

Η δουλειά περιορίστηκε αυστηρά στο `baselines/bm25/` και στο authorized
`configs/baselines_registry.yaml` update.

---

## 9. Handoff Signature

- Adapter version: **0.1.0**
- Contract version: **v2** (frozen)
- Native-fidelity: **PAPER_EXACT_NATIVE** (SciFact verified)
- Graduation commit: **62a0a0d** on `restructure/v2`
- Total files: 20 (1632 insertions)
- Disk footprint: code ~200 KB, evidence ~285 MB (gitignored)
- External cache: 2.5 GB on `D:\rag_work\pyserini_cache`

**Recommendation:** BM25 baseline είναι production-ready. Το μόνο pending
είναι re-verification στο SAH-RAG dataset (εξαρτάται από external decision)
και ablation reference (on demand). Και τα δύο είναι **zero-code-change**.

---

**Υπογραφή:** BM25 Baseline Developer
**Ημερομηνία:** 2026-10-23
