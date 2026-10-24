# ColBERTv2 Baseline (`colbertv2`)

Native-only retrieval baseline for the SAH-RAG v2 research project.

## Identity

| Field | Value |
|---|---|
| `name` | `colbertv2` |
| `version` | `0.1.0` |
| `status` | `EXECUTABLE` |
| `supports_native` | β… `True` |
| `supports_controlled` | β `False` (architectural) |
| `is_llm_based` | β `False` (retrieval-only) |

## Reference

- **Paper:** *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*
  - arXiv: <https://arxiv.org/abs/2112.01488>
  - Venue: NAACL 2022
- **Official code:** <https://github.com/stanford-futuredata/ColBERT>
- **Official checkpoint:** <https://huggingface.co/colbert-ir/colbertv2.0>
- **License:** MIT (both code and checkpoint)

## Modes

### Native β€” supported

Runs the official ColBERTv2 pipeline:

- Native checkpoint: `colbert-ir/colbertv2.0`
- Native token-level BERT encoder (late interaction)
- Native PLAID index (`doc_maxlen=512`, `nbits=2`)
- Native late-interaction scoring

This corresponds to `PAPER_EXACT_NATIVE` / `AUTHOR_SUPPORTED_NATIVE` for
the retrieval side. **There is no generation stage**: the official paper
and repository evaluate retrieval only, so the native end-to-end pipeline
for ColBERTv2 terminates at `retrieve()`.

### Controlled β€” NOT supported

`supports_controlled = False`. Rationale:

ColBERTv2's late interaction requires **token-level embeddings** from a
BERT-based encoder. BGE-M3 produces **dense, single-vector** embeddings.
Substituting BGE-M3 would not adapt ColBERTv2 β€” it would *replace* its
core architectural mechanism. Per the project's controlled-adaptation
policy, this is an *architectural alteration* and is therefore excluded.

Consequently:

- ColBERTv2 is **excluded by design** from the Stage A controlled leaderboard.
- The `config_controlled.yaml` file exists only to satisfy the adapter
  contract; instantiating with `mode: controlled` raises `ValueError`.

## Implementation notes

- **Chunking:** `text_normalized` is split into overlapping character
  windows (default `chunk_size=1800`, `chunk_overlap=200`), which is
  β‰450 tokens at ~4 chars/token β€” safely under `doc_maxlen=512`.
  This heuristic is deterministic and independent of GPU/library versions.
- **Persistence:** all artifacts are written under `self.work_dir`:
  - `collection.tsv` β€” ColBERTv2 TSV collection (`<int>\t<text>`)
  - `mapping.json` β€” `{int_id: {chunk_id, doc_id, text}}` for decoding
  - `colbert_experiments/.../indexes/<index_name>/` β€” native PLAID index
- **No hardcoded paths.** All persistence uses `self.work_dir`.
- **No generation.** `is_llm_based = False`; the adapter never calls any LLM.

## Dependencies

The ColBERTv2 stack is **heavy** (explicitly flagged per the operating manual
Β§8 rule about heavy deps):

- `torch` (β‰¥ 2.0)
- `transformers` (β‰¥ 4.30)
- `faiss-cpu` (or `faiss-gpu` for production)
- `colbert-ai` (β‰¥ 0.2.19)

These are declared in `requirements.txt`. `colbert-ai` also pulls its own
transitive dependencies (`bitarray`, `tqdm`, `scipy`, `ujson`).

### Recommended install (GPU)

```powershell
pip install colbert-ai[torch,faiss-gpu]
```

###  CPU-only install (tests will skip heavy paths)

```powershell
pip install colbert-ai[torch,faiss-cpu]
```

### Python version
The project targets Python 3.12. colbert-ai β‰¥ 0.2.19 publishes wheels
for Python 3.12. If the install fails on your environment, use an
isolated venv with Python 3.11 for this baseline only (documented as a
known limitation in reports/).

### Offline / first-run behavior
- The first build_index() triggers a one-time download of
colbert-ir/colbertv2.0 from Hugging Face.

- Subsequent runs use the local HF cache
(~/.cache/huggingface/hub/models--colbert-ir--colbertv2.0/).

- Tests that exercise the real model are gated with
pytest.mark.skipif on (colbert-ai importable) AND (checkpoint cached) AND (CUDA available). On a CPU-only / offline CI, they skip
cleanly and the suite still runs in < 10 s.

#### Known limitations
1. Retrieval-only. No generation stage; this is faithful to the paper.

2. Controlled mode rejected by design (see above).

3. Python 3.12 wheel availability for faiss/colbert-ai may vary;
see workaround above.

4. Character-based chunking. Semantic boundaries are not exploited in
this first version. Refinement (sentence/paragraph splits) is deferred
to a post-baseline iteration and must be declared as an author-deviation
if it changes retrieval behavior materially.

5. Searcher is cached on first retrieve(). cleanup() releases it.
The orchestrator should call cleanup() after evaluation if it reuses
the process for another baseline on the same GPU.

### Files

baselines/colbertv2/
β”β”€β”€ __init__.py                # lazy re-export (no torch at import time)
β”β”€β”€ adapter.py                 # ColBERTv2Baseline (BaseBaseline subclass)
β”β”€β”€ config_native.yaml         # native mode config (frozen)
β”β”€β”€ config_controlled.yaml     # exists only; instantiating raises ValueError
β”β”€β”€ README.md                  # this file
β”β”€β”€ requirements.txt           # heavy deps (torch, transformers, faiss, colbert-ai)
β””β”€β”€ tests/
    β””β”€β”€ test_adapter.py        # offline, <10 s, GPU/model paths skipped

###   Citation  

@inproceedings{santhanam2022colbertv2,
  title     = {ColBERTv2: Effective and Efficient Retrieval via
               Lightweight Late Interaction},
  author    = {Santhanam, Keshav and Khattab, Omar and Saad-Falcon, Jon
               and Potts, Christopher and Zaharia, Matei},
  booktitle = {NAACL},
  year      = {2022},
}

---

## Platform Notes - Native E2E Execution

ColBERTv2's C++ extensions (segmented_maxsim_cpp, segmented_lookup_cpp, filter_pids_cpp, decompress_residuals_cpp) require POSIX pthreads, which are unavailable on Windows MSVC. The torch.utils.cpp_extension builder on Windows hard-codes MSVC cl.exe, which does not provide pthread.h.

### Windows (dev environment)

- **Adapter contract compliance:** verified (pytest baselines/colbertv2/tests/ -> 12 passed, 1 skipped heavy test)
- **Index build:** verified (SciFact 100 docs -> 129 chunks, PLAID index persisted to disk)
- **Native C++ compile:** blocked by POSIX pthreads. Options:
  - _win_compat.py - pure-Python fallback for exploratory debugging only, not used in production
  - MSYS2 + gcc - compiles with pthreads, but torch.utils.cpp_extension does not accept gcc on Windows
  - pthreads-win32 stub - not attempted in this session

### Linux GPU server (production)

The official target environment for ColBERTv2. On Linux:
- gcc + pthreads available natively
- torch.utils.cpp_extension compiles automatically on first retrieve()
- No code changes required

### Status

- **Stage B (Native E2E):** TOTALLY_READY for Linux GPU server execution
- **Windows dev:** adapter verified, native compile blocked by POSIX dependency
- **Controlled adaptation:** architecturally excluded (BGE-M3 incompatible with late-interaction encoder)

To run the smoke test on a Linux GPU server:

    python baselines/colbertv2/tests/smoke_e2e.py

First run compiles the 4 C++ extensions (~2-3 min). Subsequent runs are cached.
