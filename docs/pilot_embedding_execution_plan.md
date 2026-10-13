# Pilot Embedding Execution Plan

**Status:** Step 82 planning freeze (no embeddings yet)
**Frozen revision:** `5617a9f61b028005a4858fdac845db406aefb181`
**Cache root:** `D:/rag_work/hf_cache` (HF_HOME-rooted, not the `hub` subdir)

## Corrections to the original plan

1. The plan initially cited revision
   `babcf60cae0a1f438d7ade582983d4ba462303c2`. That revision is not
   present on this host.
2. `.env` sets `HUGGINGFACE_HUB_CACHE=D:\rag_work\hf_cache\hub`,
   but the model was downloaded to `HF_HOME=D:\rag_work\hf_cache`
   directly. The `hub` subdirectory does not exist. Cache preflight
   reads `HF_HOME`, not `HUGGINGFACE_HUB_CACHE`.
3. License is verified from the **local** snapshot `README.md`
   (`license: mit` at line 7), not from a web page.

## Frozen model identity

| Field | Value |
|---|---|
| Model ID | `BAAI/bge-m3` |
| Revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| Revision source | local snapshot `refs/main` |
| License | MIT (local snapshot README line 7) |
| Architecture | `XLMRobertaModel` |
| Hidden size | 1024 |
| Max positions | 8194 |
| Dense dimension | 1024 |
| Dense normalized | true |
| Dense pooling | CLS |
| Sparse method | BGE-M3 learned lexical weights |
| Sparse projection | `sparse_linear.pt` |
| Multi-vector (ColBERT) | present, disabled for the pilot |

## Cache state (verified on disk)
D:/rag_work/hf_cache/
models--BAAI--bge-m3/
refs/main -> 5617a9f61b028005a4858fdac845db406aefb181
snapshots/5617a9f61b028005a4858fdac845db406aefb181/
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
sentencepiece.bpe.model
pytorch_model.bin 2,271,145,830 bytes
sparse_linear.pt
colbert_linear.pt
sentence_bert_config.json
modules.json
1_Pooling/
onnx/
README.md
imgs/

Total snapshot size: ~4.37 GB. No `.lock` files. `cache_status: VERIFIED`.

## What Step 82 does not do

- No model download
- No tokenizer load
- No weight load
- No embedding generated
- No Qdrant server started
- No vectors ingested
- No retrieval
- No evaluation
- No hosted API call
- No secrets written

## Benchmark protocol (for Step 83)

| Field | Value |
|---|---|
| Input | 12 child chunks from Step 81 |
| Batch sizes | 1, 2, 4 |
| Warmup runs | 1 |
| Measured runs per batch | 3 |
| Timer | `perf_counter` |
| Model load in timer | no |
| Persistence in timer | no |
| Embedding only in timer | yes |

Required metrics: total seconds, records/sec, ms/record, peak RSS,
dense vector count, sparse vector count. Aggregation: min, median,
max.

**A linear extrapolation from pilot seconds to real datasets is
prohibited** without an explicit token-count adjustment.

## Colab transfer (protocol only, not approved)

Local to Colab: `child_chunks.jsonl`, request manifest, checksums.
Colab to local: embedding shards, catalog, execution manifest,
checksums.

Prohibited in transfer: environment files, API keys, access tokens,
source archives, foreign code, model weights, generated answers.

Colab runs **embedding only**. No Qdrant, no retrieval, no evaluation
on Colab.

## Execution gate after Step 82
cache_inspection_approved: true (verified)
model_download_approved: false (not needed)
embedding_execution_approved: false (Step 83 will set)
index_execution_approved: false
retrieval_execution_approved: false
evaluation_execution_approved: false
generation_execution_approved: false

## Next step: Step 83

**Bounded BGE-M3 CPU Embedding Benchmark.** Runs the local model on
the 12 child chunks, measures batch sizes 1/2/4, three measured runs
each, produces embedding shards, catalog, manifest, checksums. Still
no Qdrant, no retrieval, no evaluation.

Because the frozen snapshot is verified on disk, Step 83 does not
need a controlled download step. It goes straight to the benchmark.
