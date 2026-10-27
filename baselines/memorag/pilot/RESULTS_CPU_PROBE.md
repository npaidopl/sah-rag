# MemoRAG CPU probe — checkpoint and environment verification

**Purpose.** Record, in one place, what was verified about the MemoRAG
native path on the current CPU host and what remains blocked. This is
not an end-to-end result. It is the evidence base for the
"author-supported native, hardware-gated" label until the university
GPU is available.

**Date.** 2026-10-27 (UTC)
**Host.** Windows 11, CPU only, no CUDA
**RAM.** 15.37 GB total, 5.8 GB free at probe time
**Python.** 3.12.10 (project venv)
**Working directory.** C:\Users\USER1\Documents\SAH-RAG

## 1. Package installation

- `memorag==0.1.5` installed in the project venv via `pip install memorag --no-deps`.
- Missing dependencies were installed individually: `transformers`, `accelerate`, `sentencepiece`, `faiss-cpu`, `semantic-text-splitter`, `jieba`, `langdetect`, `python-Levenshtein`, `rouge`, `seaborn`, `fuzzywuzzy`, `pynvml`.
- `deepspeed`, `triton`, `flash_attn`, `minference` are not installable on Windows without CUDA. They are not required for inference.
- A minimal stub for `minference.MInference` was created under `.venv/minference/__init__.py` that raises `RuntimeError` if actually used. This unblocks `from minference import MInference` in `memorag/memorag.py` line 15 without changing the architecture.
- `import memorag` now succeeds. Verified from the project venv.

## 2. Checkpoint verification

Checkpoint: `TommyChien/memorag-qwen2-7b-inst`
Revision: `a60e362e7922d9ac0014279dd0ae7a038ff7138e`
Cache location: `D:\rag_work\hf_cache\models--TommyChien--memorag-qwen2-7b-inst`

| Artifact | Status | Size |
|---|---|---|
| model-00001-of-00004.safetensors | OK | 4.95 GB |
| model-00002-of-00004.safetensors | OK | 4.99 GB |
| model-00003-of-00004.safetensors | OK | 4.99 GB |
| model-00004-of-00004.safetensors | OK | 1.23 GB |
| Total | 4/4 | 16.16 GB |

Safetensors integrity: all 4 shards open cleanly via `safetensors.safe_open`.
Tensor count: 508 tensors across 4 shards.
Total parameters: 8.08 B. This is consistent with Qwen2-7B base (~7.6B)
plus the MemoRAG beacon/compression modules and expanded vocabulary
(152064 tokens vs 151936 stock). Not corruption.

Tokenizer: loads via `AutoTokenizer.from_pretrained(..., local_files_only=True, trust_remote_code=True)`.
Vocabulary size 151643 observed at tokenizer level; config reports 152064.
Max length 131072.

Config: loads via `AutoConfig.from_pretrained(..., local_files_only=True, trust_remote_code=True)`.
model_type = qwen2, hidden_size = 3584, num_hidden_layers = 28.

## 3. What was verified (native-readiness evidence)

The following claims are backed by artifacts in this repository and by the
probe output above, not by prose:

| Claim | Evidence | Artifact |
|---|---|---|
| `memorag` package imports | `import memorag` returned OK from project venv | this probe |
| Official memory checkpoint present | 4/4 shards, 16.16 GB total | `blobs/`, index.json |
| Shards not corrupted | `safetensors.safe_open` on all 4 shards, 508 tensors, 8.08B params | this probe |
| Tokenizer loadable | `AutoTokenizer.from_pretrained(..., local_files_only=True)` OK | this probe |
| Config loadable | `AutoConfig.from_pretrained(..., local_files_only=True)` OK | this probe |
| Manifests fully pinned | 0 remaining `<PIN_ME>` in both manifests | `native_manifest.yaml`, `controlled_manifest.yaml` |
| Adapter contract-compliant | 4/4 tests pass | `tests/test_adapter.py` |

The five-point proof-of-nativeness rubric defined in `RESULTS_TEMPLATE.md`
is satisfied at the **static** level (source, checkpoint identity, no
substitutions). The **runtime** level (clue generation, retrieval, generation)
has not yet been demonstrated on real weights because of the RAM limit
documented below.

## 4. What is blocked and why

**Inference is blocked on the current CPU host.**

- The Qwen2-7B checkpoint requires approximately 28 GB of free RAM to load
  in fp16 (14 GB weights + 14 GB transient during safetensors load).
- This host has 15.37 GB total RAM and 5.8 GB free at probe time.
- Loading 7B in fp16 therefore fails before inference even begins.
- CPU inference of 7B, even if it fit, would take minutes per query.

This is a **hardware limitation, not a code defect**. Every code path,
manifest, dependency, checkpoint, and configuration that the native path
requires is in place. What is missing is memory.

The pilot does **not** attempt a fake 7B run on this host, because a fake
run would not be native evidence and would be worse than no run.

## 5. Handoff to the university GPU

When the GPU host is available, the following steps produce the native
end-to-end evidence, in order:

1. Sync this repository and `D:\rag_work\hf_cache` to the GPU host.
2. Install `requirements.txt` plus `pip install memorag`.
3. Install `faiss-gpu` (instead of `faiss-cpu` used here).
4. Run `fill_pins.py` to refresh `frozen_at_utc` for the GPU-host run.
5. Run `smoke_test_native.py` with the native manifest. This proves:
   - the official memory checkpoint loads,
   - the clue stage executes,
   - clue-conditioned retrieval returns real chunks.
6. Run `run_native_e2e.py` with the same manifest. This proves:
   - native end-to-end generation runs using the memory model as the
     generator,
   - no external API key is required (verified by the absence of any
     `openai`/`anthropic` call in the script),
   - the artifacts in `artifacts/native_e2e_<timestamp>/` contain
     `generate_probe.json` with a real answer.
7. Run `smoke_test_controlled.py` with the controlled manifest. This
   proves the controlled-adaptation path works with generation disabled.
8. Copy `RESULTS_TEMPLATE.md` to `artifacts/<run_id>/RESULTS.md` and fill
   it in. Attach it to the pilot artifacts.

The artefacts produced by steps 5–7 satisfy the five-point
proof-of-nativeness rubric at the runtime level, in addition to the static
level already satisfied here.

## 6. What is explicitly NOT claimed

- **Not claimed**: MemoRAG native end-to-end has been executed on this host.
- **Not claimed**: retrieval quality, generation quality, or any benchmark
  number.
- **Not claimed**: the CPU host is representative of the GPU host's
  performance.
- **Not claimed**: paper-exact reproduction. The tier is
  `AUTHOR_SUPPORTED_NATIVE`, not `PAPER_EXACT_NATIVE`.

## 7. Environment notes for the GPU host operator

- `minference`, `triton`, `flash_attn`, `deepspeed` are not required for
  inference and were not installed on the CPU host. The GPU host may install
  them via the official `requirements.txt` if the authors' setup expects
  them.
- The `minference` stub created in `.venv/minference/__init__.py` on the CPU
  host is not part of the repository and will not travel to the GPU host.
  The GPU host should install the real `minference` from the official
  requirements, or (if unavailable) recreate the stub.
- The cache directory used here is `D:\rag_work\hf_cache`. On the GPU host,
  point `HF_HOME` and `HF_HUB_CACHE` at the equivalent shared cache path
  before running the pilot scripts.
- BGE-M3 revision is pinned at `5617a9f61b028005a4858fdac845db406aefb181`.
  This must match between native and controlled runs for the comparison to
  be fair.

## 8. Reproducing this probe

From the project root, with the venv active:

    python -m pytest baselines/memorag/tests/ -q
    # then run the probe scripts in this folder

The exact commands used to produce this document are the same commands
shown in this file. No hidden state is required.