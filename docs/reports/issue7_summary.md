# Issue 7 - Qwen model downloads (14B + 32B)

**Status:** DONE
**Date closed:** 2026-11-02

## Purpose

Provide the two Qwen2.5 models required by the controlled-mode protocol:
- Qwen2.5-14B-Instruct for indoor LLM (RAG_INDOOR_LLM_ROOT)
- Qwen2.5-32B-Instruct for generation LLM (fallback to 14B documented)

Both verified via transformers AutoConfig + AutoTokenizer with local_files_only=True.

## Qwen2.5-14B-Instruct

| Item | Value |
|---|---|
| Path | D:\rag_work\models\Qwen2.5-14B-Instruct |
| Size | 27.52 GB |
| Shards | 8/8 |
| Tensors | 579 |
| hidden_size | 5120 |
| num_hidden_layers | 48 |
| vocab_size | 151643 |
| model_type | qwen2 |

Notes:
- Downloaded incrementally over ~13.5h due to bandwidth limits (~830 KiB/s peak)
- huggingface-cli 0.24.7 failed repeatedly (HTTP fallback, presigned URL expiry)
- Resolved with aria2c -x 16 -c segmented download
- Orphan .incomplete files cleaned (~260 MB freed)

---

## Qwen2.5-32B-Instruct

| Item | Value |
|---|---|
| Path | D:\rag_work\models\Qwen2.5-32B-Instruct |
| Size | 61.04 GB |
| Shards | 17/17 |
| Tensors | 771 |
| hidden_size | 5120 |
| num_hidden_layers | 64 |
| vocab_size | 151643 |
| model_type | qwen2 |

Notes:
- Download split across two machines:
  - PC (~14.55 GB, wifi, ~0.73 MB/s)
  - Laptop (~46.49 GB, LTE, ~3.79 MB/s) - 5x faster
- Total laptop session: 3h 13m for 42.86 GB
- hf CLI + aria2c with outer retry loop (200 attempts, fresh URL per attempt)
- All .aria2 control files cleaned after completion

---

## Download infrastructure

Script: scripts/download_qwen_32b.ps1 (resumable, portable across machines)

Key design decisions:
1. Metadata source: https://huggingface.co/api/models/<id>/tree/main?recursive=true
   (the simple /api/models/<id> endpoint does NOT return size per file).
2. aria2c flags: -c -x 16 -s 16 -k 1M (resume, 16 connections, 16 segments, 1MB min split).
3. Outer retry loop: 200 attempts per file, 30s wait, fresh URL per attempt
   (avoids HF presigned URL expiry at ~4h).
4. Checkpointing: .safetensors.aria2 control files + partial .safetensors
   stored next to target. Moving the SSD moves the checkpoints.
5. Logging: each event to <target>/_download_progress.log (append-only,
   survives restarts, records host/timestamp/progress).

Usage:
    powershell -ExecutionPolicy Bypass -File scripts\download_qwen_32b.ps1

Idempotent: re-running skips complete shards, resumes partials.
If all complete: prints ALL FILES ALREADY COMPLETE.

---

## Environment (persistent, User scope)

| Variable | Value |
|---|---|
| HF_HOME | D:\rag_work\hf_cache |
| HF_HUB_CACHE | D:\rag_work\hf_cache\hub |
| TORCH_HOME | D:\rag_work\torch_cache |

.env (project root, gitignored):
- SAH_RAG_MODELS_DIR=D:\rag_work\models
- RAG_INDOOR_LLM_ROOT=D:\rag_work\models\Qwen2.5-14B-Instruct

## MANIFEST

D:\rag_work\models\MANIFEST.md - authoritative record:
- Qwen2.5-14B-Instruct: 27.52 GB, 8/8 shards, verified 2026-11-01
- Qwen2.5-32B-Instruct: 61.04 GB, 17/17 shards, verified 2026-11-02
- bge-large-en-v1.5: pending Issue 6
- Deferred: multi-qa-mpnet (blocked on xlm_rst_tuned), NV-Embed-v2 (CC-BY-NC)

---

## Verification checklist

- [x] 14B: 8/8 shards, byte-exact sizes
- [x] 14B: 579 safetensors tensors
- [x] 14B: config + tokenizer load
- [x] 32B: 17/17 shards, byte-exact sizes
- [x] 32B: 771 safetensors tensors
- [x] 32B: config + tokenizer load
- [x] No stale .aria2 files
- [x] MANIFEST.md updated
- [x] Download script committed to scripts/

## Lessons learned

1. HF API: /api/models/<id> returns siblings without size. Use /tree/main?recursive=true.
2. aria2c + outer retry: for 403 URL expiry, an external PowerShell loop
   with fresh URL beats --max-tries inside aria2c.
3. Stale .aria2: after graceful shutdown, control files may remain on
   complete shards. Clean after verifying shard is complete.
4. Machine portability: script works from any machine with PowerShell + aria2c.
   Checkpointing is fully filesystem-based (no host dependency).
5. Bandwidth != Throughput: for large downloads, stability (dedicated LTE)
   beats shared wifi even when raw bandwidth looks similar.

## Related docs

- docs/evaluation_boundary.md - v2 metrics (Issue 2)
- docs/llm_fairness_policy.md - will be created in Issue 1
- D:\rag_work\models\MANIFEST.md - authoritative model list
- LaTeX reports (in paper draft, not in this repo): issue7_qwen14b_download.tex, issue7_32b_download_complete.tex

