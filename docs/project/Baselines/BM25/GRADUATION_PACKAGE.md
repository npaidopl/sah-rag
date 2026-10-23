# BM25 Baseline β€” Graduation Package

**Date:** 2026-10-22 23:01:20
**For:** SAH-RAG supervisor
**Status:** Ready for supervisor review and graduation decision.

## Executive summary

The BM25 baseline adapter is functionally complete and verified:

- SciFact (5,183 docs, 300 queries): **PAPER_EXACT_NATIVE** β€” 300/300 queries
  match official top-10 by docid AND score (max delta = 0.000000).
- MS MARCO passage v1 (8.8M docs, 6,980 queries): MRR@10 = 0.187486 vs
  published 0.187 (delta = 0.00049, PASS).
- Controlled-mode equivalence: Native and controlled produce identical
  retrieval output; verified by automated test.

All work is contained within baselines/bm25/. No files outside this
directory have been modified.

## Files in workspace

| File | Purpose |
|---|---|
| adapter.py | BM25Baseline (subclasses frozen BaseBaseline) |
| config_native.yaml | Native-mode config (k1=0.9, b=0.4) |
| config_controlled.yaml | Controlled-mode config |
| README.md | Description, license, deps, known limitations |
| ARM_REGISTRY.md | Arm IDs, native-fidelity classification, scope |
| DATASET_SELECTION.md | Dataset decision record |
| GRADUATION_PACKAGE.md | This file |
| requirements.txt | pyserini==2.4.0 (Java 21 system requirement) |
| tests/conftest.py | Faulthandler monkeypatch (Windows+JVM) |
| tests/test_adapter.py | Base build+retrieve test |
| tests/test_controlled_mode.py | Native = Controlled verification |

## Evidence artifacts

| Artifact | Location | Value |
|---|---|---|
| SciFact official run | runs/scifact.bm25.trec | nDCG@10 = 0.678909 (294K lines) |
| SciFact adapter run | runs/scifact.adapter_e2e.trec | nDCG@10 = 0.678909 (3K lines) |
| MS MARCO official run | runs/msmarco_passage_v1.bm25.trec | MRR@10 = 0.187486 (6.97M lines) |
| Stage 1 report | reports/stage1_scifact.md | Reproduction check (PASS) |
| Stage 2 report | reports/stage2_msmarco.md | Credibility check (PASS) |
| Adapter E2E report | reports/adapter_e2e_scifact.md | PAPER_EXACT_NATIVE confirmation |

## Verification command

    .\.venv\Scripts\Activate.ps1
    python -m pytest baselines/bm25\tests\ -v

Expected: 3 tests pass (or skip if Java 21 not on PATH).

## Deliverables status

| Deliverable | Status | Notes |
|---|---|---|
| (alpha) native E2E | VERIFIED (PAPER_EXACT_NATIVE for SciFact) | Ready for re-verification on SAH-RAG dataset |
| (beta) controlled-adaptation | VERIFIED (native = controlled) | Automated test proves equivalence |
| (gamma) ablation reference | OPTIONAL, pending SAH-RAG request | Recommendation doc when SAH-RAG team asks |

## Known limitations

1. Java 21 required (pyserini==2.4.0 + jnius cannot load JVM on Java 22+).
2. k1/b overrides recorded but not applied (Pyserini CLI uses Lucene defaults).
3. Document-level only (chunk_id == doc_id).
4. Controlled-mode injection structurally unused (BM25 is lexical, no LLM).
5. pytest faulthandler monkeypatch for Windows+JVM (conftest.py only).
6. Cold JVM startup ~10-15s (amortized in real workloads).

## Graduation recommendation

**Recommend: GRADUATE** to baselines/bm25/.

Evidence supports this:
- Reproduction verified at both small (SciFact) and large (MS MARCO) scale.
- Native-fidelity verified as PAPER_EXACT_NATIVE for SciFact.
- Controlled-mode equivalence proven by automated test.
- Contract compliance verified (no rogue imports, no paths outside workspace,
  no external API calls in tests).

## Suggested graduation steps (supervisor executes)

1. Run: python -m pytest baselines/bm25\tests\ -v
2. Review adapter.py and ARM_REGISTRY.md section 2 (structural exception).
3. Move: baselines/bm25 -> baselines/bm25
4. Add to configs/baselines_registry.yaml:

    - name: bm25
      status: executable
      supports_native: true
      supports_controlled: true
      is_llm_based: false
      notes: "Lexical baseline (Pyserini/Anserini). Native-fidelity: PAPER_EXACT_NATIVE (SciFact). Controlled-mode components structurally unused (ARM_REGISTRY section 2)."

5. Commit: feat(baselines): add bm25 adapter (Pyserini/Anserini)

## After graduation

Once BM25 is in baselines/bm25/, the orchestrator can invoke it through
BaseBaseline for both native and controlled modes. Results will flow into
SAH-RAG comparison tables as BM25_NATIVE and BM25_CONTROLLED arms (identical
by construction; single run suffices).

The (gamma) ablation reference will be produced ONLY when the SAH-RAG team
requests it, as a view of the existing native run. No new run will be
executed for (gamma).

## Environment

- Python 3.12
- Java 21 (Temurin 21.0.12.1)
- pyserini 2.4.0
- pytrec_eval_terrier 0.5.11
- Cache: D:\rag_work\pyserini_cache (2.5 GB MS MARCO), D:\rag_work\hf_cache