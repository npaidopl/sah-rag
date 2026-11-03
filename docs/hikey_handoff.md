# HiKEY Handoff — For the Next Session

**Date:** 2026-10-29
**Status:** COMPLETE (code + tests + smoke test verified)
**Commits:** 3 (see below)

---

## TL;DR

HiKEY is fully implemented across 4 artifacts. All 13 tests pass offline.
Full smoke test verified on synthetic PDFs (2 PDFs, 3 queries, real PyMuPDF
parsing). Everything is ready to run on real datasets when they arrive.
No official code exists — this is a paper-derived reimplementation, honestly
labeled as such.

---

## What the next person needs to know

1. **There is NO official HiKEY code.** This is a paper-derived reimplementation. Cite accordingly.
2. **Native pipeline requires PDFs with hierarchy.** PyMuPDF fallback works always.
3. **Real encoders (gte-Qwen2-7B + SigLIP) require GPU.** Models live on `D:\rag_work\models`.
4. **LVLM reader (Qwen2.5-VL-7B) is optional.** IR works without it.
5. **Do NOT use the controlled adapter to claim "SAH-RAG beats HiKEY".** See rule at end.

---

## Commits (in order)
61da4c3 feat(baselines): add hikey (reported) + hikey_controlled + native pipeline
1852d60 fix(hikey_native): assign real cosine scores to units; use pymupdf import
26ee608 feat(hikey_native): add --mock-encoders CLI flag
<next> fix(hikey_native): wire cache_dir param into model loaders

Branch: `restructure/v2` (local, ahead of origin by 4 commits after CELL 2)

---

## Artifacts

### 1. Reported baseline — `baselines/hikey/`

- `adapter.py` — `HiKEYReportedBaseline(ReportedBaseline)`
- `reported_metrics.yaml` — M3DocVQA / FRAMES-PDF / AVG, verbatim from paper
- `README.md` — comparison scope rule

**What it exposes:** `get_reported_metrics(dataset_id) -> dict[str, float]`

**Registered in:** `configs/baselines_registry.yaml` as `hikey` (status: reported)

### 2. Controlled adapter — `workspace/hikey_controlled_dev/`

- `adapter.py` — `HiKEYBaseline(BaseBaseline)` name=`hikey_controlled`
- 6 tests, offline
- Text-only approximation (NOT HiKEY)

**Status:** workspace-stage. Graduate to `baselines/hikey_controlled/` only after
a controlled pipeline is finalized (needs ablation decision).

### 3. Native pipeline — `workspace/hikey_native_dev/`

- `pipeline.py` — `HiKEYNativePipeline` (standalone, not BaseBaseline)
- `run.py` — CLI entry point
- `components/` — dhp_parser, card_builder, multimodal_encoder, maxsim,
  evidence_assembler, lvlm_reader
- 7 tests, offline

**Status:** committed. Ready for real PDFs.

### 4. Ablation proposal — `docs/hikey_ablation_proposal.md`

3 candidates: MaxSim, Upper Context, Ancestry-aware evidence assembly.
Recommendation: start with MaxSim.
---

## How to verify (offline, ~2s)

    Set-Location "C:\Users\USER1\Documents\SAH-RAG"
    & ".\.venv\Scripts\Activate.ps1"
    python -m pytest workspace/hikey_native_dev/tests/ workspace/hikey_controlled_dev/tests/ -q
    # Expected: 13 passed in ~1.2s

---

## How to run

### Smoke test (no models, no GPU) - verified working

    python -m workspace.hikey_native_dev.run `
        --corpus workspace\hikey_smoke_test\corpus `
        --queries workspace\hikey_smoke_test\queries.json `
        --out workspace\hikey_smoke_test\out.jsonl `
        --mock-encoders

### Real run (requires GPU + models on D:)

    # First session: set env vars (see Environment section)
    python -m workspace.hikey_native_dev.run `
        --corpus <real_pdf_dir> `
        --queries <real_queries.json> `
        --out <output.jsonl>

### With LVLM answers (optional)

    python -m workspace.hikey_native_dev.run `
        --corpus <real_pdf_dir> `
        --queries <queries.json> `
        --out <output.jsonl> `
        --answer

---

## Environment setup

### D:\rag_work layout (already populated)

    D:\rag_work\
    ├── models\
    │   ├── bge-m3          # already downloaded
    │   └── Qwen*           # already downloaded (Qwen2.5-VL or similar)
    ├── hf_cache\           # HF_HOME target
    │   └── transformers\
    └── torch_cache\        # TORCH_HOME target

**Already present:** bge-m3 and Qwen models are in D:\rag_work.
The next session should verify exact paths before loading.

### Persistent env vars (set via CELL 2)

    HF_HOME            = D:\rag_work\hf_cache
    TRANSFORMERS_CACHE = D:\rag_work\hf_cache\transformers
    TORCH_HOME         = D:\rag_work\torch_cache

These are User-scope env vars - they persist across PowerShell sessions.
The config.yaml also declares cache_dir: d:/rag_work/models (used by
MultiModalEncoder when passed explicitly).

### .venv

Stays on C:\Users\USER1\Documents\SAH-RAG\.venv\. Do NOT move -
paths are hardcoded by pip on Windows.

### Dependencies for real execution (GPU)

    pip install torch>=2.0 transformers>=4.45 FlagEmbedding>=1.3
    pip install Pillow openai
    # Optional for better DHP:
    pip install doclayout-yolo

These are commented out in requirements.txt - uncomment when needed.

---

## LVLM reader modes

Three options in components/lvlm_reader.py:

1. Mock (default for tests) - LVLMReader(mock=True)
2. Local Qwen2.5-VL-7B - LVLMReader(mock=False, api_mode=False) (~15GB VRAM)
3. API mode - LVLMReader(mock=False, api_mode=True) + env vars HIKEY_LVLM_API_KEY, HIKEY_LVLM_API_BASE

When neither local nor API is available, answer() returns
[LVLM_UNAVAILABLE] and retrieve() works normally.

---

## DHP backends (priority order)

1. m3docdep - exact paper module (not currently installable)
2. doclayout-yolo - layout detection
3. PyMuPDF heuristic - always works, current default

To add M3DocDep: implement _try_m3docdep() in components/dhp_parser.py.
The hook is already there - just returns None until implemented.

---

## What's pending (external, not blocking execution)

| Item | Blocks | Notes |
|---|---|---|
| Datasets | real runs | 7 curated + future additions. PDFs with tables/figures ideal. |
| GPU | real encoders | gte-Qwen2-7B ~14GB, SigLIP ~1GB. University GPU access planned. |
| M3DocDep checkpoint | better DHP | PyMuPDF fallback works. Hook exists in dhp_parser.py. |
| API key | API LVLM mode only | Local Qwen2.5-VL doesn't need it. |

Nothing in the code is blocking execution. When datasets arrive, just run.

---

## Known limitations (be honest in papers)

1. Paper-derived, not official. No public repo exists. Cite as
   "paper-derived reimplementation of HiKEY (Shin et al., 2026)".
2. Mock embeddings produce random rankings. Real encoders required
   for meaningful metrics. Smoke test only verifies plumbing.
3. DHP via PyMuPDF is font-heuristic, not learned. May miss complex layouts.
4. LVLM not evaluated end-to-end yet. IR-only was the focus of this session.
5. HiKEY was only evaluated on M3DocVQA and FRAMES-PDF. Do NOT compare
   to SAH-RAG on other datasets using paper numbers.

---

## The not-HiKEY rule (CRITICAL for papers)

The controlled adapter (workspace/hikey_controlled_dev/) is a
text-only approximation. It is NOT HiKEY. In any paper:

- CORRECT: "SAH-RAG vs HiKEY on M3DocVQA (paper-reported)"
- CORRECT: "SAH-RAG vs text-only hierarchical retriever (HiKEY-inspired) on our datasets"
- WRONG: "SAH-RAG beats HiKEY on [any dataset other than M3DocVQA/FRAMES]"

Every chunk from the controlled adapter carries
metadata["approximation"] = "hiKEY_text_only_approximation" - a hard marker.

The native pipeline (workspace/hikey_native_dev/) can legitimately claim
"paper-derived HiKEY reimplementation" - but ONLY on datasets where the
full multimodal pipeline runs (M3DocVQA, FRAMES-PDF, or equivalent).

---

## Suggested next steps

1. Smoke test on droplet dataset - when a PDF corpus arrives, run native pipeline.
2. Install real encoders - gte-Qwen2-7B + SigLIP on D: (bge-m3 already present).
3. Evaluate on M3DocVQA/FRAMES-PDF - if native comparison is desired.
4. Implement ablation Candidate 1 (MaxSim) - see docs/hikey_ablation_proposal.md.
5. Consider next baseline - same methodology: native + controlled + reported,
   with smoke test at the end.

---

## Files you will touch first

| Task | File |
|---|---|
| Debug parsing | workspace/hikey_native_dev/components/dhp_parser.py |
| Adjust scoring weights | workspace/hikey_native_dev/config.yaml |
| Real encoders | workspace/hikey_native_dev/components/multimodal_encoder.py |
| LVLM | workspace/hikey_native_dev/components/lvlm_reader.py |
| Add M3DocDep | _try_m3docdep() in dhp_parser.py |
| Run smoke | workspace/hikey_smoke_test/ |
| Test | workspace/hikey_native_dev/tests/test_pipeline.py |

---

## Contact / context

- Paper: https://arxiv.org/abs/2605.29606
- Project page: https://gyuhoshim.github.io/projects/hikey/
- ACL: 2026.acl-long.818

---

## Verification checklist for the next session

- [ ] python -m pytest workspace/hikey_native_dev/tests/ workspace/hikey_controlled_dev/tests/ -q -> 13 passed
- [ ] python -m workspace.hikey_native_dev.run --help -> shows --mock-encoders flag
- [ ] D:\rag_work\models\ has bge-m3 and Qwen (already present)
- [ ] Env vars set: $env:HF_HOME, $env:TRANSFORMERS_CACHE, $env:TORCH_HOME
- [ ] Read docs/hikey_ablation_proposal.md if continuing SAH-RAG work

---

## Position in SAH-RAG v2 architecture

The native pipeline (workspace/hikey_native_dev/) is NOT a BaseBaseline.
It runs standalone, not through the orchestrator. Its outputs are treated as
"author-supported native reimplementation" - parallel evidence to paper-reported
numbers, not a substitute.

When the orchestrator runs `hikey` in the manifest, it reads reported_metrics.yaml
(paper numbers). To compare against native reimplementation, run manually:

    python -m workspace.hikey_native_dev.run --corpus <pdf_dir> --queries <queries.json>

and integrate results manually into the ablation section of the paper.

---

## D: paths - known inconsistency (for next session)

Current state (audited 2026-10-29):

- D:\rag_work\           ACTIVE (26GB hf_cache, 2.5GB models, 5GB venvs, parsers)
- D:\sah-rag\            EMPTY (data/, experiments/, models/ all empty skeletons)
- SAH-RAG/.env points to D:\sah-rag\* (stale paths)
- HF_HOME env var points to D:\rag_work\hf_cache (correct, matches actual data)

The .env file is FROZEN per organizer manual rule 3 - do NOT modify it in this session.
Next session decision: consolidate D:\rag_work\* -> D:\sah-rag\ OR update .env to
point to D:\rag_work\. Recommendation: update .env (D:\rag_work is where everything lives).

Verification needed by next session:
- Confirm exact path of bge-m3 (may be nested deeper than depth-4 search)
- Confirm Qwen model location
- Decide consolidation strategy before GPU sync

---

## Decisions (locked in this session)

1. Reported baseline: graduated, no further action.
2. Controlled adapter: graduated to baselines/hikey_controlled/ as separate entry.
   Paper positioning: FOOTNOTE (not separate row in comparison table).
3. Native pipeline: standalone, position documented above.
4. D: paths: keep separate for now, .env untouched (frozen). Decision for next session.
5. Ablation proposal: documentation only, not part of v2 IR comparison. Future work.

---

## FRAMES-PDF rendering (added 2026-11-04)

The FRAMES dataset is text-only (2,521 Wikipedia articles in plain text
form). The paper rendered each article to PDF via Playwright + Chromium.
We reproduced this workflow.

### What we did

1. Fetched live Wikipedia HTML via Playwright (not the local wikitext,
   which has no structure - 0 headers, 0 tables, 0 images)
2. Rendered 2,521 articles to PDF in 36.8 min (3 parallel workers)
3. Verified: 293 avg headers/PDF (from 1 in v1 attempt)

### Location

- Renderer: ``scripts/frames_render/render_from_wikipedia.py``
- Prep script: ``scripts/frames_render/prepare_smoke.py``
- Output PDFs: ``D:\rag_work\raw_datasets\frames\pdfs\``
  (2,521 PDFs, 1.9 GB, avg 293 headers/PDF)
- Old v1 (plaintext): ``D:\rag_work\raw_datasets\frames\pdfs_v1_plaintext\``
  (deprecated, do not use for native evaluation)

### Why v1 failed

v1 rendered the ``articles.jsonl`` wikitext directly. The text has no
HTML structure: 0 ``== headers ==``, 0 ``{| tables |}``, 0 images across
all 100 sampled articles. The output PDFs had 1 header each (the title).

v2 fetched live HTML which has 15-20 h2, 10-40 h3, 1-28 tables, 3-8
figures per article. This is the correct approach.

### Verification

    python scripts\frames_render\prepare_smoke.py --limit 20
    python -m workspace.hikey_native_dev.run `
        --corpus workspace\hikey_smoke_test\frames_corpus `
        --queries workspace\hikey_smoke_test\frames_queries.json `
        --out workspace\hikey_smoke_test\frames_out.jsonl `
        --mock-encoders

Expected: 20/20 queries, 2-5 chunks each, 0 empty, 0 numeric-only,
section_title present in metadata, gold_articles present in output.

---

## Known limitations (updated 2026-11-04 after FRAMES-PDF work)

The following limitations are **known** and **documented** as honest
disclosures for papers. They do NOT block execution.

1. **DHP via PyMuPDF is font-heuristic, not learned.** May mis-classify
   infobox table cells as text units (e.g., 'yttrium <-', 'BCNOFN').
   Real M3DocDep parser will fix this. Hook exists in
   ``components/dhp_parser.py::_try_m3docdep()``.

2. **Mock embeddings produce random rankings.** The smoke tests confirm
   plumbing, not retrieval quality. Real gte-Qwen2-7B is required for
   meaningful metrics.

3. **FRAMES-PDF is from live Wikipedia (2026-11-04 snapshot), not the
   paper's 2024-09-19 snapshot.** Articles may have drifted. For
   strict reproduction, use the ``revision_id`` field in
   ``articles.jsonl`` to fetch historical revisions.

4. **CUAD legal PDFs are flat** (1 section per PDF). This is expected:
   contracts don't have Wikipedia-style hierarchy. Used for smoke test
   only, not for native HiKEY evaluation.

5. **LVLM reader not tested end-to-end.** IR-only was the focus.

---

## Verification checklist (updated 2026-11-04)

Run these before touching anything:

    # 1. Git state
    cd C:\Users\USER1\Documents\SAH-RAG
    git log --oneline -10
    # Expected: aa64229 ... 61da4c3 (10+ commits)

    # 2. Tests
    .\.venv\Scripts\Activate.ps1
    python -m pytest workspace\hikey_native_dev\tests\ -q
    # Expected: 7 passed

    # 3. Registry
    python -c "import yaml; [print(b['name'], b['status']) for b in yaml.safe_load(open('configs/baselines_registry.yaml'))['baselines'] if 'hikey' in b['name']]"
    # Expected:
    #   hikey reported
    #   hikey_controlled executable

    # 4. FRAMES-PDF corpus
    (Get-ChildItem D:\rag_work\raw_datasets\frames\pdfs\*.pdf).Count
    # Expected: 2521

    # 5. Env vars
    Write-Host $env:HF_HOME
    # Expected: D:\rag_work\hf_cache

    # 6. FRAMES smoke test
    python scripts\frames_render\prepare_smoke.py --limit 20
    python -m workspace.hikey_native_dev.run `
        --corpus workspace\hikey_smoke_test\frames_corpus `
        --queries workspace\hikey_smoke_test\frames_queries.json `
        --out workspace\hikey_smoke_test\frames_out.jsonl `
        --mock-encoders
    # Expected: 20 queries, 76/77 PDFs matched, 89 chunks, 0 empty

## What to do when GPU arrives

1. ``pip install torch transformers FlagEmbedding``
2. Download models to ``D:\rag_work\models\``:
   - ``Alibaba-NLP/gte-Qwen2-7B-instruct`` (~14 GB)
   - ``google/siglip-large-patch16-384`` (~1 GB)
3. Re-run native pipeline WITHOUT ``--mock-encoders``:
       python -m workspace.hikey_native_dev.run `
           --corpus D:\rag_work\raw_datasets\frames\pdfs `
           --queries workspace\hikey_smoke_test\frames_queries_full.json `
           --out experiments\hikey_frames_full.jsonl `
           --top-k 10
4. Compare against paper-reported FRAMES-PDF metrics:
   - EM: 10.5, ANLS: 14.6, ROUGE-L: 17.7, METEOR: 15.4

## What to do when Kaggle M3DocVQA arrives

1. Download full M3DocVQA (~6 GB, 3,300 PDFs)
2. Place at ``D:\rag_work\raw_datasets\m3docvqa\pdfs\``
3. Re-run native pipeline on that directory
4. Compare against paper-reported M3DocVQA metrics:
   - EM: 27.5, ANLS: 30.7, ROUGE-L: 32.2, METEOR: 23.9
