# Provenance of prompt templates

All template files in this directory are derived from files at pinned
commit `778454e2986c39f2630fd8832abb44a5c9bc124f` of
https://github.com/TIGER-AI-Lab/LongRAG (MIT license).

## Source file mapping

| Template | Source | Source lines |
|---|---|---|
| `nq.txt` | `utils/gpt_inference.py` | `predict_nq`, lines 57-62 |
| `hotpotqa.txt` | `utils/gpt_inference.py` | `predict_hotpotqa`, lines 68-75 |
| `answer_extraction.txt` | `utils/gpt_inference.py` | `extract_answer`, lines 97-102 (static prefix) |
| `answer_extraction_demos.txt` | `utils/gpt_inference.py` | `generate_demo_examples`, lines 80-91 (from HF) |

## Verifying a template against the pinned commit

```powershell
# 1. Extract blob content from pinned commit
git -C external\baselines\longrag cat-file -p "778454e:utils/gpt_inference.py" > tmp_gpt_inference.py

# 2. Inspect the relevant method
Get-Content tmp_gpt_inference.py | Select-Object -Skip 54 -First 10  # predict_nq
```

## Why placeholder substitution instead of literal f-strings

The official code uses Python f-strings inside method bodies, which
mixes code logic with prompt text. We transcribe only the literal text
into `.txt` templates with named placeholders (`{context}`,
`{question}`, `{titles}`, `{long_answer}`, `{demos}`), allowing the
runner to compose them from structured inputs. The rendered output is
byte-identical to what the official code produces for the same input
values, except that Python's `str.format` of lists differs slightly
from f-string list interpolation — both produce the same repr, so the
final strings are identical.

## Known deviations

- The `post_process` step (extracting after `"answer:"`) is applied
  by the reader implementations, not the prompt templates. This
  matches the official code path (`GPTInference.predict` calls
  `self.post_process` internally).
- The demos are cached on disk instead of re-fetched from HF on each
  run. The SHA-256 of the cached demos is recorded in the manifest.