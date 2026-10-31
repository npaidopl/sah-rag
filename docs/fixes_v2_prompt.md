# SAH-RAG v2 - Fixes Session v2 (final, self-contained)

## Role
Software engineer on SAH-RAG v2 research project. Execute 5 issues in order. All decisions are pre-made. Do NOT ask for clarifications. If you hit genuine ambiguity, note it in the final report and proceed with best judgment.

## Environment
- Project root: `C:\Users\USER1\Documents\SAH-RAG`
- Venv: `.venv\Scripts\Activate.ps1` (Python 3.12)
- Env vars set: `HF_HOME=D:\rag_work\hf_cache`, `TORCH_HOME=D:\rag_work\torch_cache`, `HF_HUB_CACHE=D:\rag_work\hf_cache\hub`
- Disk: 440GB free on D:

## Context - ALREADY DONE (do not touch)
- Issue 3 (`datasets/` shadowing): DONE via rename to `sah_datasets/`
- Issue 4 (`.env` paths): DONE, consolidated to `D:\rag_work`
- Issue 5 (controlled dispatch): DONE, `evaluation/runner.py:_build_controlled_components` exists
- Test baseline: 219 failed, 921 passed, 3 skipped, 10 errors (in `_baseline_tests_before.txt`)
- 15 baselines: 12 executable, 3 reported (in `configs/baselines_registry.yaml`)

## Remaining issues (MANDATORY ORDER: 2 -> 8 -> 7 -> 6 -> 1)

---
### Issue 2 - Evaluation Boundary (code + doc)

Code changes:

evaluation/retrieval_metrics.py:
- Keep existing recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k unchanged
- ADD function hit_at_k(retrieved_doc_ids, gt_doc_ids, k) that returns 1.0 if any gt doc in top-k, else 0.0 (return 0.0 if not gt_doc_ids or k<=0)
- Add hit_at_k to __all__

evaluation/runner.py:
- Change CUTOFFS = [1, 3, 5, 10] to CUTOFFS = [1, 5, 10]
- Import hit_at_k alongside existing metrics
- In buckets init loop: add buckets[f"hit@{k}"] = []
- In accumulation loop: add buckets[f"hit@{k}"].append(hit_at_k(retrieved_ids, gt_ids, k))
- Add field metric_version: "v2" to output metrics.json dict

Doc - create docs/evaluation_boundary.md:
- Metrics: Recall@{1,5,10}, MRR@10, nDCG@10, Hit@{1,5,10}
- Level: document-level
- Averaging: macro + micro
- Exclusions: queries with empty GT excluded (report count)
- Splits: test if available, else full
- Reproducibility: seed=42, deterministic tie-breaks
- Metric version: v2 (v1 was [1,3,5,10] without hit@k; macro averages NOT comparable across versions)

Doc - create baseline_audit/evaluation_boundary.yaml with these keys:
  version: 2.0
  metric_version: v2
  cutoffs: [1, 5, 10]
  metrics: [recall, mrr, ndcg, hit]
  level: document
  averaging: [macro, micro]
  exclusions: empty_gt
  splits: test_else_full
  seed: 42

Downstream audit BEFORE changing code - run:
  Get-ChildItem . -Recurse -Include *.py,*.md,*.yaml | Select-String -Pattern "1,\s*3,\s*5,\s*10|hit@3|CUTOFFS"
If a consumer of metrics.json needs more than 5 lines to adapt: STOP, report file path + required change + production/legacy flag.

Acceptance:
- python -c "from evaluation.retrieval_metrics import hit_at_k; print('OK')" works
- python -c "from evaluation.runner import CUTOFFS; print(CUTOFFS)" prints [1, 5, 10]
- pass_count >= 921 after changes

---

### Issue 8 - QwenLLM Option B+ (no default, runtime guard)

sah_rag/components/llms.py changes:
- Remove default for model_path (make it required or None)
- If model_path is None at init: raise ValueError('QwenLLM requires explicit model_path. Controlled protocol mandates Qwen2.5-14B-Instruct.')
- If '14B' not in str(model_path) and strict_fairness=True: raise ValueError(f'Controlled protocol mandates Qwen2.5-14B-Instruct. Got: {model_path}. Pass strict_fairness=False for tests/debugging.')
- Add parameter strict_fairness: bool = True
- Update docstring: 'Wrapper for Qwen2.5-14B-Instruct. Used exclusively in controlled-mode injection.'

evaluation/runner.py _build_controlled_components changes:
- Pass model_path explicitly from config: llm = QwenLLM(model_path=entry['config_controlled'].get('indoor_llm_path'), strict_fairness=True)
- If config doesn't have indoor_llm_path, raise clear error (do not fallback silently)

Caller audit BEFORE changing - run:
  Get-ChildItem sah_rag, baselines, evaluation, orchestration, tests -Recurse -Filter *.py | Select-String -Pattern 'QwenLLM\('
If >5 callers: update ALL with explicit model_path. If <=3: may use strict_fairness=False in tests only, never production.

Acceptance:
- QwenLLM() without model_path raises ValueError
- QwenLLM(model_path='.../Qwen2.5-1.5B', strict_fairness=True) raises ValueError
- QwenLLM(model_path='.../Qwen2.5-14B', strict_fairness=True) works (config only, no load)
- pass_count >= 921 after changes

---

### Issue 7 - Qwen 14B + 32B downloads

Pre-check - run:
  Get-PSDrive D | Select-Object Free
Must be >= 200 GB free. Otherwise STOP and report.

Download (huggingface-cli, has resume):
  huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir D:\rag_work\models\Qwen2.5-14B-Instruct
  huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir D:\rag_work\models\Qwen2.5-32B-Instruct

Verify 14B (config + tokenizer only, no model load):
  python -c "from transformers import AutoConfig, AutoTokenizer; p=r'D:\rag_work\models\Qwen2.5-14B-Instruct'; c=AutoConfig.from_pretrained(p); t=AutoTokenizer.from_pretrained(p); print(f'14B OK: hidden={c.hidden_size}, layers={c.num_hidden_layers}, vocab={t.vocab_size}')"

Verify 32B (size + config only, NO load - 64GB won't fit in RAM):
  Get-ChildItem D:\rag_work\models\Qwen2.5-32B-Instruct -Recurse -File | Measure-Object Length -Sum
  (Get-ChildItem D:\rag_work\models\Qwen2.5-32B-Instruct -Filter '*.safetensors').Count
  python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained(r'D:\rag_work\models\Qwen2.5-32B-Instruct'); print(f'32B config OK: hidden={c.hidden_size}, layers={c.num_hidden_layers}')"

Expected: 14B ~28GB, 32B ~64GB. If a download fails 3 times: BLOCKED and report.

Create D:\rag_work\models\MANIFEST.md with a markdown table:
  columns: Model | Path | Size | Revision | Verified
  rows: Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct, bge-large-en-v1.5 (from Issue 6)

Acceptance:
- Both dirs exist with config.json, *.safetensors, tokenizer.json
- MANIFEST.md created

---

### Issue 6 - bge-large-en-v1.5 only

Download:
  huggingface-cli download BAAI/bge-large-en-v1.5 --local-dir D:\rag_work\models\bge-large-en-v1.5

Update MANIFEST.md with bge-large-en-v1.5 row.

DEFER (do NOT download):
- sentence-transformers/multi-qa-mpnet-base-cos-v1: only for DISRetrieval native, which is BLOCKED on missing xlm_rst_tuned backbone. Note in MANIFEST: 'Deferred: needed only for DISRetrieval native (blocked)'
- nvidia/NV-Embed-v2: CC-BY-NC license, only for HippoRAG2 paper-exact (author-supported uses BGE-M3, already have). Note in MANIFEST: 'Deferred: CC-BY-NC license; only for paper-exact'

Acceptance:
- bge-large-en-v1.5 exists
- Deferred models documented in MANIFEST

---

### Issue 1 - Fairness policy (doc + yaml)

Create docs/llm_fairness_policy.md with 4 sections:

Section 1 - Native E2E Rules:
- Each baseline uses its own native LLM
- If paid and no free author-supported: excluded
- Table of all 15 baselines from configs/baselines_registry.yaml (columns: name, indoor_llm, generation_llm, native_e2e)

Section 2 - Controlled Protocol:
- Qwen2.5-14B-Instruct (indoor) + Qwen2.5-32B-Instruct (generation)
- Shared across all is_llm_based=True baselines
- Include VERBATIM this fallback policy:
  'If Qwen2.5-32B-Instruct is unavailable at evaluation time, generation falls back to Qwen2.5-14B-Instruct. This MUST be (a) recorded in each run metrics.json under generation_model field, (b) flagged in any comparison table, and (c) disclosed in the paper as a limitation.'

Section 3 - SAH-RAG Generation:
- Qwen2.5-14B-Instruct, standard RAG prompt
- Note: 'temporary (Project 2 will replace)'

Section 4 - Exemptions & Rationale:
- List baselines excluded from native E2E with reason (paid API, license, missing components)

Create baseline_audit/llm_fairness_policy.yaml with these top-level keys:
  version: '1.0'
  created_at: '2026-10-31'
  native_e2e_interpretation: (explain Y / reported / N / N-A semantics)
  native_e2e: {rules: [...], baselines: {<exact_name>: {status, indoor, generation, native_e2e}}}
  controlled: {indoor_llm, generation_llm, generation_llm_fallback, applies_to}
  sah_rag_generation: {llm, mode}
  exemptions: {<exact_name>: reason}

CRITICAL for yaml: use EXACT name values from configs/baselines_registry.yaml. Do not abbreviate. If registry has both 'hikey' (reported) and 'hikey_controlled' (executable), include BOTH.

Steps for LLM:
1. Run: python -c "import yaml; r=yaml.safe_load(open('configs/baselines_registry.yaml', encoding='utf-8')); [print(b['name'], b.get('status'), b.get('is_llm_based','?')) for b in r['baselines']]"
2. Fill native_e2e.baselines with EXACT names from that output - no abbreviations, no omissions
3. Fill exemptions for every baseline with native_e2e: 'N'

Acceptance:
- Both files exist, YAML parses
- Every registry baseline appears exactly once in native_e2e.baselines
- Number of exemptions == number of native_e2e: 'N' entries

---

## Hard rules (violating any = STOP)

- Do NOT modify baselines/base.py, docs/adapter_contract.md, or any frozen contract
- Do NOT use paid APIs
- Do NOT store models on C:
- After EVERY issue run: python -m pytest tests/ -q --no-header --tb=no 2>&1 | Select-Object -Last 1
- If pass_count < 921 after any issue: STOP, report, wait
- If a downstream consumer needs >5 lines: STOP, report
- Order is mandatory: 2 -> 8 -> 7 -> 6 -> 1

## Final report - create docs/fixes_v2_report.md with these sections:

Summary: markdown table with columns Issue | Status | Notes. Rows: 2, 8, 7, 6, 1. Status one of DONE/PARTIAL/BLOCKED.

Tests: Before (219 failed / 921 passed / 3 skipped / 10 errors), After (fill).

QwenLLM callers: count + list of call sites.

Downstream consumers of metrics.json: files updated for CUTOFFS change; files that would need changes but STOPPED.

Model manifest: Qwen2.5-14B-Instruct (path, size, verification), Qwen2.5-32B-Instruct (path, size, verification), bge-large-en-v1.5 (path, size), Deferred (multi-qa-mpnet reason, NV-Embed-v2 reason).

Blockers: any or 'none'.

Commit: hash + message.

## Final instruction
Execute autonomously. After each issue, post the pytest summary line. At the end, produce docs/fixes_v2_report.md, commit with message 'fix(fixes-v2): evaluation boundary, QwenLLM guard, model downloads', and post final summary + hash. If any issue is BLOCKED, state it and continue only if the next issue is independent.

END OF PROMPT
