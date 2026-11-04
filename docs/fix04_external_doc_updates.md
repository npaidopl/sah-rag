# FIX-04 — External Documentation Updates

**Status:** User action required (external .tex file)

**Date:** 2026-11-04

## Purpose

`baselines_updates.tex` is an external document (not in the SAH-RAG repo)
that tracks baseline readiness. It needs updates to reflect:

1. **DISRetrieval** — controlled track is fully verified E2E
2. **HiKEY** — native smoke test passed on FRAMES-PDF (2,521 PDFs)
3. **HiKEY** — 7 commits now pushed to origin

This file provides **copy-paste-ready snippets** for the user to
insert into their local `baselines_updates.tex`.

## Section 1: DISRetrieval Update

**Location in .tex:** Section "Baselines που Περιμένουν Hardware/Data (Documentation Only)"

**BEFORE (current text):**

    DISRetrieval
    - Τι λείπει: xlm_rst_tuned/ backbone (445 MB)
    - Γιατί δεν μπορεί τώρα: Δεν διανέμεται από τους authors
    - Ενέργεια: Contact authors (GitHub issue / email). Αν αποκτηθεί, ο κώδικας είναι έτοιμος.
    - Controlled: ✅ Έτοιμο (αναμονή GPU)

**AFTER (new text):**

    DISRetrieval
    - Controlled: ✅ TOTALLY_READY (verified E2E: 2 docs, 3 queries,
                   scores 0.75/0.58/0.55, 12 tests passing, 3× runner tests)
    - Native: 🟡 Code complete (paper-exact), BLOCKED on missing xlm_rst_tuned/
              backbone (445 MB, not distributed by authors)
    - Native API key: ✅ NOT REQUIRED (vLLM config is free)
    - Ενέργεια controlled: Τρέχει όταν υπάρξει GPU + datasets
    - Ενέργεια native: (προαιρετικά) contact authors για backbone
    - Σημείωση: Το native ΔΕΝ απαιτεί paid API key. Μόνο το backbone λείπει.

## Section 2: HiKEY Update

**Location in .tex:** Section "Baselines που Περιμένουν Hardware/Data (Documentation Only)"

**BEFORE (if exists):**

    HiKEY
    - Τι λείπει: PDF datasets + GPU + models
    - Γιατί δεν μπορεί τώρα: Data modality audit required
    - Controlled: ✅ Έτοιμο

**AFTER (new text):**

    HiKEY
    - Reported: ✅ TOTALLY_READY (paper metrics in reported_metrics.yaml)
    - Controlled: ✅ TOTALLY_READY (registered as hikey_controlled)
    - Native: 🟡 READY_FOR_GPU (standalone pipeline, smoke-tested on FRAMES-PDF:
              2,521 PDFs, 76/77 gold PDFs matched, 89 chunks, 0 noise)
    - Ενέργεια: GPU + gte-Qwen2-7B + SigLIP (όταν έρθει GPU)
    - ΣΗΜΑΝΤΙΚΟ: Το native pipeline ΔΕΝ είναι BaseBaseline (standalone),
                 δεν συμμετέχει στο orchestration. Θα τρέξει ξεχωριστά.
    - FRAMES-PDF artifacts: D:/rag_work/raw_datasets/frames/pdfs/ (2,521 PDFs)
    - CUAD smoke: 5 PDFs (flat hierarchy expected for contracts)
    - Handoff: docs/hikey_handoff.md (443 lines, 24 sections)

## Section 3: HiKEY Git Status

**No .tex change needed.** Confirmed via:

    git log --oneline -20 | Select-String "hikey|frames"
    → 98bcc77 docs(hikey): update handoff with FRAMES-PDF rendering outcomes
    → 55854ad fix(hikey_native): expose section_title in metadata + pass through gold_articles
    → aa64229 feat(frames): add FRAMES smoke test preparation script
    → f618576 fix(hikey_native): add missing 'import re' in pipeline.py
    → 0500c73 fix(hikey_native): filter numeric-only and empty units in evidence assembly
    → 4f47ae4 feat(frames): render FRAMES articles to PDF from live Wikipedia HTML
    → e6a5b46 fix(hikey_native): PyMuPDF text normalization + noise filtering

All pushed to origin/restructure/v2.

## Section 4: What Still Needs Updating (Future Sessions)

These will be handled when the corresponding fixes complete:

| Section | Baseline | When |
|---|---|---|
| FIX-05 | Multi-Meta-RAG (controlled runner) | After FIX-05 |
| FIX-06 | LightRAG (controlled E2E) | After FIX-06 |
| FIX-07 | 17 datasets migration | After FIX-07 |
| FIX-08 | LLM fairness policy finalize | After FIX-08 |

## Section 5: How to Apply

1. Open `baselines_updates.tex` in your LaTeX editor
2. Find the "Baselines που Περιμένουν Hardware/Data" section
3. Locate the DISRetrieval entry — replace with the new text above
4. Locate the HiKEY entry — replace with the new text above
5. Save the file
6. Recompile with `xelatex baselines_updates.tex` (twice for TOC)

**No git commit needed** for `baselines_updates.tex` — it lives outside
the repo. Keep it with your paper drafts.

## Change log

- **2026-11-04** (FIX-04): created this document.