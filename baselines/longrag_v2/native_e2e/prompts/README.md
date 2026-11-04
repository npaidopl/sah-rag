# Prompt templates (native-faithful)

Byte-faithful transcriptions from `utils/gpt_inference.py` at pinned
commit `778454e2986c39f2630fd8832abb44a5c9bc124f`
(TIGER-AI-Lab/LongRAG, MIT license).

## Two-step generation flow

The LongRAG reader is a **two-step pipeline**, matching the official
`predict_nq` / `predict_hotpotqa` + `extract_answer` methods:

**Step 1 (main prompt)** — produces `long_answer`:

| File | Placeholders | Datasets |
|---|---|---|
| `nq.txt` | `{titles}`, `{context}`, `{question}` | NQ |
| `hotpotqa.txt` | `{titles}`, `{context}`, `{question}` | HotpotQA |

**Step 2 (extraction prompt)** — produces `short_answer` (scored):

| File | Placeholders | Used by |
|---|---|---|
| `answer_extraction.txt` | `{demos}`, `{question}`, `{long_answer}` | Both |
| `answer_extraction_demos.txt` | (no placeholders — fixed 8-shot demos from HF) | Both |

The `short_answer` from Step 2 is the quantity scored against ground
truth via `eval_util.single_ans_em` (exact match).

## Native dataset support

Only **NQ** and **HotpotQA** are supported as native. This matches the
official repository, which implements `predict_nq` and
`predict_hotpotqa` only — no `predict_qasper` or
`predict_multifieldqa` exists.

Any attempt to run Qasper or MultiFieldQA-en via this pipeline would
be a **controlled adaptation**, not a native claim. The paper
evaluated those two datasets via an external LongBench harness that
is not part of the TIGER-AI-Lab/LongRAG repository.

## Byte-faithfulness notes

The template files preserve several quirks of the official f-strings:

- `nq.txt` ends with `". "` (period + space), no trailing newline.
- `hotpotqa.txt` ends with `"thing.  "` (two spaces), no trailing newline.
- `hotpotqa.txt` contains `{question}.For general` — no space after
  the period; preserved verbatim.
- `answer_extraction.txt` contains `words.It's` — missing space after
  the period (typo in official code); preserved verbatim.

The `test_prompt_loader.py` suite asserts all of these quirks.

## Prompt hash verification

The SHA-256 of each file is computed at load time by
`native_e2e/prompt_loader.py` and recorded in every run manifest.
To verify the current on-disk SHA-256:

```powershell
Get-FileHash baselines\longrag_v2\native_e2e\prompts\nq.txt -Algorithm SHA256
```

## Demo examples

`answer_extraction_demos.txt` is fetched from HuggingFace
(`TIGER-Lab/LongRAG`, subset `answer_extract_example`, 8 examples)
and cached to disk. To regenerate:

```python
from baselines.longrag_v2.native_e2e.prompt_loader import regenerate_demos
regenerate_demos(num_demo=8)
```

## Deleting non-native files

Files `qasper.txt` and `multifieldqa.txt` were deleted during Phase 2
of the native-fidelity work (see `baseline_audit/longrag_native_*.yaml`).
They contained transcriptions from the paper appendix, not from the
official repository, and could not have supported a native claim.