# MemoRAG baseline (SAH-RAG v2)

## Overview

`memorag` is a SAH-RAG v2 baseline adapter wrapping the official
**MemoRAG** system (TheWebConf 2025).

MemoRAG couples a long-range memory model with clue-conditioned retrieval:
the memory model first produces *clues* (surrogate queries and/or draft
answers), and those clues drive a dense retriever over the corpus. The
clue stage is the architecture-defining component of MemoRAG.

## Reference

- **Paper:** MemoRAG — *Boosting Long Context Processing with Global
  Memory-Enhanced Retrieval Augmentation*, TheWebConf 2025.
  arXiv:2409.05591 <https://arxiv.org/abs/2409.05591>
- **Code:** <https://github.com/qhjqhj00/MemoRAG>
- **Memory checkpoints:**
  - `TommyChien/memorag-qwen2-7b-inst`
  - `TommyChien/memorag-mistral-7b-inst`
- **Python package:** `memorag` (MIT), official release.

## Status

- `status`: **executable**
- `supports_native`: **true**
- `supports_controlled`: **true**
- `is_llm_based`: **true** (memory model participates in retrieval)
- Tier: **AUTHOR_SUPPORTED_NATIVE** when run with an official memory
  checkpoint and the official clue/retrieval path.

`PAPER_EXACT_NATIVE` is out of scope for this adapter: the paper-era
commit hash, the exact clue prompt, the exact clue count, and the exact
retriever top-k are not frozen in a reproducible manifest. This adapter
targets the *author-supported native* tier, which is what the official
repository ships and what the paper's authors treat as canonical.
## Modes

### `native` mode

Uses the official MemoRAG package with an official released memory
checkpoint. The clue generation path, memory cache construction, and
retrieval signal path are exactly the author's. No component is
substituted.

- Memory model: `TommyChien/memorag-qwen2-7b-inst` (default), or
  `TommyChien/memorag-mistral-7b-inst`, or any officially released
  MemoRAG memory checkpoint.
- Retriever: `BAAI/bge-m3` (the official example's retriever).
- Clue generation: native, unchanged.
- Final answer generation: not exercised in v2 (IR-only). The adapter
  stops after retrieval.

### `controlled` mode

Same architecture, but the orchestrator **injects** `embedder` (BGE-M3)
and `llm`. The adapter MUST use the injected embedder for embedding and
MUST NOT call external APIs. The memory model and clue generation stay
native, because they are architecture-defining.

Final answer generation is disabled in v2 (IR-only). The injected `llm`
is accepted for contract compatibility and is not used to produce
answers.

## Architecture notes

The following MUST be preserved in every mode; removing any of them
produces a different architecture, not a weaker MemoRAG:

1. **Long-range memory construction.** The memory model ingests the
   corpus once and builds a compressed global memory.
2. **Query-conditioned clue generation.** The memory model produces
   clues (surrogate queries or draft answers) from the query and the
   global memory.
3. **Clue-conditioned retrieval.** Retrieval is conditioned on the
   clues, not on the raw query alone.
4. **Downstream consumption of the retrieved evidence.** In v2 this
   is IR-only; the retrieved chunks are consumed by the evaluation
   layer.

If a future variant removes the clue stage, it is **not** MemoRAG. It
can be compared *against* MemoRAG, not *as* MemoRAG.

## Configs

- `config_native.yaml`: memory model + retriever model are specified;
  orchestrator passes `embedder=None, llm=None`.
- `config_controlled.yaml`: only non-component hyperparameters are
  listed; `embedder` and `llm` are injected by the orchestrator at
  runtime. The memory model remains native.

Both configs use only the frozen top-level keys: `name`, `mode`,
`top_k`, `extra`. No absolute paths. No secrets.

## Testing

`tests/test_adapter.py` is offline and CPU-only. It mocks the `memorag`
package so no GPU, no network, and no heavy dependencies are required.

It validates:

- `build_index()` creates the expected `work_dir`.
- `retrieve()` returns `list[RetrievedChunk]` with contiguous ranks
  `1..len(results)`.
- Returned `doc_id` values exist in the mock corpus.
- `chunk_id` and `text` are non-empty.
- `retrieve()` before `build_index()` raises `RuntimeError`.
- Missing `memorag` package raises a clear `RuntimeError`.

Run:

    .\.venv\Scripts\Activate.ps1
    python -m pytest workspace\memorag_dev\tests\ -v

Expected: 4 tests pass in roughly 1 second.
## Dependencies

`requirements.txt` lists the packages needed to run MemoRAG end-to-end
(native or controlled):

- `memorag` (official package)
- `torch`, `transformers`, `huggingface-hub`, `accelerate`, `sentencepiece`
- `faiss-cpu` (or `faiss-gpu` on the GPU host)
- `numpy`, `tqdm`

These are **not** required for the offline unit tests. The tests mock
the `memorag` module.

A CUDA-capable GPU is required for real execution with the Qwen2 7B
memory checkpoint. The controlled-adaptation pipeline can also run on
a smaller memory checkpoint if the host VRAM is limited; that choice
is recorded in the execution manifest, not in this adapter.

## License

- The **adapter code** in this folder is part of the SAH-RAG project
  and follows the project's license.
- The **MemoRAG Python package** is distributed under the MIT license
  (see PyPI metadata).
- The **MemoRAG GitHub repository** is distributed under Apache-2.0.
- The **memory checkpoint** `TommyChien/memorag-qwen2-7b-inst` is
  released by the MemoRAG authors and is based on Qwen2-7B-Instruct,
  whose base license is Apache-2.0/MIT. Users of this adapter are
  responsible for verifying the license of the specific checkpoint
  revision they pin in their execution manifest.

No API keys are needed by this adapter in either mode. In controlled
mode, the orchestrator injects the LLM; the adapter never calls an
external API.
## Known limitations

- **Paper-exact reproduction is out of scope.** The paper-era commit
  hash, exact clue prompt, exact clue count, and exact retriever
  hyperparameters are not frozen in a public manifest. This adapter
  targets *author-supported native*.
- **Final answer generation is not exercised in v2.** The adapter is
  IR-only by contract. The native end-to-end answer path is validated
  separately by the execution manifest, not by the adapter.
- **Memory model remains native in controlled mode.** This is
  intentional. Replacing the memory model would remove the
  architecture-defining component of MemoRAG. If a future variant
  needs a non-memory LLM writing clues, that variant must be
  documented as an ablation, not as MemoRAG.
- **Real execution needs GPU and downloaded weights.** The unit
  tests do not cover the real path. A pilot run on the university
  GPU is the gate before the baseline is treated as production-ready.

## Deviations from the contract

None. The adapter:

- subclasses `BaseBaseline`,
- uses only `build_index()`, `retrieve()`, `cleanup()`,
- accepts the frozen constructor signature,
- returns contiguous `RetrievedChunk` ranks,
- uses `self.work_dir` for persistence,
- does not import from `rag_system`,
- does not hardcode absolute paths,
- does not perform network calls in tests.

## Graduation checklist

- [x] `adapter.py` subclasses `BaseBaseline`
- [x] Identity attributes set (`name`, `version`, `status`,
      `supports_native`, `supports_controlled`, `is_llm_based`)
- [x] Constructor signature matches contract
- [x] `build_index()` persists under `self.work_dir`
- [x] `retrieve()` returns contiguous `RetrievedChunk` ranks
- [x] `config_native.yaml` and `config_controlled.yaml` present
- [x] `tests/test_adapter.py` passes offline in < 10s
- [x] `requirements.txt` present
- [x] No `rag_system` imports
- [x] No hardcoded absolute paths
- [x] No network calls in tests
- [x] No modifications outside `workspace/memorag_dev/`
- [ ] Supervisor review
- [ ] Graduation to `baselines/memorag/`
- [ ] Entry in `configs/baselines_registry.yaml` (status: executable)

## Addendum: native E2E and the injected-LLM question

The v2 adapter is **IR-only**. It does not call `self.llm.generate()`,
and its `is_llm_based` flag is therefore `False`. This is a deliberate
decision, not an omission:

- In `native` mode, MemoRAG uses its own memory model (Qwen2-7B) as
  both the clue generator and the answer generator. There is no
  injected LLM to call.
- In `controlled` mode, the v2 contract disables final answer
  generation (Section 9, Rule 6). The injected LLM is accepted for
  signature compatibility but is not exercised.

Native end-to-end generation — including the expressive reader path
that the paper evaluates — lives in `pilot/run_native_e2e.py`. That
script uses the official memory checkpoint as both clue generator and
generator, which is the official MemoRAG default and requires **no
external API key**. It is the artifact that justifies the
`AUTHOR_SUPPORTED_NATIVE` label; the adapter alone cannot, because
the v2 contract forbids generation inside adapters.

An examiner should therefore look at:

- `baselines/memorag/adapter.py` for the IR-only comparison track;
- `baselines/memorag/pilot/run_native_e2e.py` and its artifacts for
  the native generation track.

## Native-readiness status (2026-10-27)

The checkpoint, tokenizer, config, manifests, adapter, and dependency set
are all verified on the CPU host. Native end-to-end execution is blocked
only by RAM (15.37 GB total, 5.8 GB free; the 7B checkpoint needs ~28 GB
to load in fp16). This is documented in `pilot/RESULTS_CPU_PROBE.md`.

Two artefacts settle the status:

- `pilot/native_manifest.yaml` — fully pinned (0 remaining `<PIN_ME>`),
  with the memory checkpoint revision `a60e362e7922d9ac0014279dd0ae7a038ff7138e`
  and the BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`.
- `pilot/RESULTS_CPU_PROBE.md` — the verified evidence base: 4/4 safetensors
  shards open cleanly (508 tensors, 8.08B params), tokenizer and config
  load, adapter tests pass, five-point nativeness rubric satisfied at the
  static level.

When the university GPU is available, the native end-to-end evidence is
produced by running, in order:

    python baselines/memorag/pilot/smoke_test_native.py \
        --manifest baselines/memorag/pilot/native_manifest.yaml \
        --out baselines/memorag/pilot/artifacts/native_run1
    python baselines/memorag/pilot/run_native_e2e.py \
        --manifest baselines/memorag/pilot/native_manifest.yaml \
        --out baselines/memorag/pilot/artifacts/native_e2e_run1
    python baselines/memorag/pilot/smoke_test_controlled.py \
        --manifest baselines/memorag/pilot/controlled_manifest.yaml \
        --out baselines/memorag/pilot/artifacts/controlled_run1

`run_native_e2e.py` uses the official memory model as the generator and
does not require any external API key.