# MemoRAG pilot — native + controlled

This folder holds the pilot tooling that must run **before** any dataset
is attached to the baseline. The pilot proves, on a tiny corpus, that:

1. the native end-to-end MemoRAG path executes on the GPU host, and
2. the controlled-adaptation path executes under the v2 contract.

The pilot is deliberately small: 5 documents and 1 query. Its purpose
is not to measure quality; it is to prove that the real code path runs,
that the checkpoint loads, that the clue stage executes, and that
artifacts are produced.

## Contents

| File | Purpose |
|---|---|
| `native_manifest.yaml` | Frozen record of the native configuration |
| `controlled_manifest.yaml` | Frozen record of the controlled configuration |
| `smoke_test_native.py` | Real (non-mock) end-to-end native probe |
| `smoke_test_controlled.py` | Real (non-mock) controlled probe |
| `artifacts/` | One subdirectory per run, timestamped |

Both manifests use `<PIN_ME>` markers for fields that can only be
filled in on the GPU host (commit hash, checkpoint revision, VRAM
budget, generator identity). Both smoke tests refuse to run while
`<PIN_ME>` markers remain in critical fields.

## Order of operations

1. Provision the GPU host and install `requirements.txt` plus the
   official MemoRAG package at the pinned commit.
2. Fill in `<PIN_ME>` in `native_manifest.yaml`:
   - `source.commit` — run `git rev-parse HEAD` in the MemoRAG clone
   - `models.memory.revision` — the HF revision returned by
     `huggingface_hub.snapshot_download`
   - `models.retriever.revision` — the BGE-M3 revision under
     `D:/rag_work/hf_cache`
   - `models.generator.enabled` — `true` for full e2e, `false` for
     IR-only (the pilot defaults to `true`)
   - `models.generator.id` / `revision` if `enabled: true`
   - `context.max_tokens` — measured against actual VRAM
   - `context.clue_count` — native default from the package
   - `execution.min_vram_gb` — measured
3. Run the native smoke test:

       python workspace/memorag_dev/pilot/smoke_test_native.py `
         --manifest workspace/memorag_dev/pilot/native_manifest.yaml `
         --out workspace/memorag_dev/pilot/artifacts/native_run1

4. Fill in `<PIN_ME>` in `controlled_manifest.yaml` (same fields plus
   `models.injected_llm.id` / `revision`).
5. Run the controlled smoke test:

       python workspace/memorag_dev/pilot/smoke_test_controlled.py `
         --manifest workspace/memorag_dev/pilot/controlled_manifest.yaml `
         --out workspace/memorag_dev/pilot/artifacts/controlled_run1

6. Record outcomes in `artifacts/RESULTS.md`.
## Expected artifacts per run

Each run directory contains:

| File | Meaning |
|---|---|
| `env.txt` | Python, platform, torch version, CUDA devices, VRAM |
| `probe.json` | Whether `memorag` imports; which top-level symbols exist |
| `methods.json` | Live introspection of the pipeline object (native only) |
| `load.log` | Step-by-step log of the run |
| `retrieve_probe.json` | Query, top_k, elapsed time, raw output |
| `generate_probe.json` | Only present if generation was enabled (native) |
| `run.json` | Final status record |

If `probe.json` reports `imported: false`, the pilot cannot proceed;
the GPU host is missing the `memorag` package or a dependency. If
`retrieve_probe.json` reports `raw: null` or an empty list, the
pipeline loaded but did not produce retrievals; that is a real signal,
not a mock artifact.

## What counts as proof of native-ness

For an examiner to verify that the implementation is native, the
artifacts must collectively show:

1. **Exact source.** `probe.json` records `package_file` and `version`.
   `env.txt` records the Python executable and working directory. The
   manifest records the commit hash and model revisions.
2. **Real checkpoint.** `load.log` shows the memory model identifier
   passed to `MemoRAG(...)` and the elapsed load time. A 7B checkpoint
   loading in less than a few seconds is impossible; the log therefore
   proves the real checkpoint was used.
3. **Clue stage executed.** `retrieve_probe.json` contains output that
   is produced only after clue generation, since retrieval in MemoRAG
   is clue-conditioned. If `raw` is empty, the pilot failed, not passed.
4. **Generation on/off is explicit.** For native full e2e, the manifest
   has `generator.enabled: true` and `generate_probe.json` exists. For
   controlled, the manifest has `generation.enabled: false` and
   `generate_probe.json` must be absent.
5. **No substitutions.** For native mode, neither the manifest nor the
   log shows an injected embedder or LLM. For controlled mode, the
   only injected component is the embedder; the memory model stays
   native and the log records that it does.

These five points are what an examiner can verify directly from the
artifacts. If any is missing, the pilot is not yet ready to justify
a native claim in a publication.

## Failure handling

The smoke tests exit with codes that distinguish the failure class:

| Code | Meaning | What to fix |
|---|---|---|
| 0 | Success | Nothing |
| 1 | Missing dependency | Install missing package |
| 2 | Manifest still `<PIN_ME>` | Fill in the field |
| 3 | No CUDA device | Use a machine with CUDA |
| 4 | Runtime failure | Inspect `load.log` and `methods.json` |

Exit code 4 is the interesting case: it usually means the
`pipeline(...)` signature does not match the one assumed by the smoke
test. In that case:

1. Open `methods.json` and inspect `callables`.
2. Adjust the smoke test to match the actual signature.
3. Do **not** silently fall back to a mock. The point of the pilot is
   to run the real code.

## Anti-drift rule

If any of the following changes between the native pilot and the
production run, a new pilot must be executed first:

- memory model identifier or revision,
- retriever identifier or revision,
- generator identifier or revision,
- commit of the MemorAG source,
- clue count or clue prompt (if exposed by the package),
- context budget.

The pilot is the gate. A green pilot is what allows a dataset to be
attached. If the pilot goes red after any of the above changes, the
run is not a native MemoRAG run anymore, even if the class name is
the same.

## Relation to the v2 adapter

The v2 adapter in `workspace/memorag_dev/adapter.py` is IR-only by
contract. The pilot is the place where the real native end-to-end
path (including generation) is proven. They are complementary:

- The **adapter** is what the SAH-RAG evaluation harness invokes.
- The **pilot** is what justifies the label `AUTHOR_SUPPORTED_NATIVE`.

Both must pass before the baseline is promoted to `baselines/memorag/`.

## What this pilot does NOT do

- It does not measure quality on real data.
- It does not compare against SAH-RAG.
- It does not produce a leaderboard.
- It does not replace the controlled-adaptation evaluation that will
  run once datasets are frozen.

Those are downstream phases. The pilot is the gate that says "the
code path runs"; nothing more, nothing less.