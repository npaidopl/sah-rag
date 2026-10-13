# Seven Real Long-Context Datasets Inventory and Sizing

**Step:** 91
**Status:** Read-only inventory
**Scope:** Infrastructure validation only. Not a publishable result.

## Purpose

Discover, inventory, and structurally size the seven real
long-context datasets without executing any processing stage:

- ruler
- loong
- long_bench_v2
- lara
- greek_bar_bench
- cuad
- clerk

No parser, chunker, tokenizer, model, embedding, Qdrant, retrieval,
evaluation, or generation is executed.

## Allowed operations

- resolve_paths
- stat_files
- compute_sha256
- stream_jsonl
- validate_json_syntax
- count_records
- inspect_record_keys
- compute_text_lengths
- compute_whitespace_token_estimates
- inspect_identity_cardinality
- inspect_reference_integrity
- estimate_derived_workload

## Prohibited operations

- rewrite_jsonl
- normalize_text
- run_adapter
- parse_documents
- generate_chunks
- load_embedding_model
- load_model_tokenizer
- create_embeddings
- create_qdrant_collection
- execute_retrieval
- execute_evaluation
- execute_generation

## Discovery rule

Datasets are searched under multiple candidate roots:

- `processed/canonical/<dataset>/`
- `data/<dataset>/`
- `datasets/<dataset>/`
- `processed/<dataset>/`

Aliases are recorded as metadata only:

- `long_bench_v2` → `longbench_v2`, `LongBenchV2`
- `greek_bar_bench` → `greekbarbench`
- `clerk` → `CLERK`

A dataset with no unique canonical root is rejected with
`status: AMBIGUOUS_ROOT` and `inventory_allowed: false`.

## Measurement units

- `bytes` (integer)
- `characters` (integer)
- `whitespace_tokens` (integer)

### Token measurement

The collector uses a whitespace-token proxy:
dense float32 bytes = N × 1024 × 4

Sparse vector storage cannot be estimated without the real model's
nonzero distribution.

## Status vocabulary

Each dataset receives one of:

- `INVENTORIED`
- `MISSING`
- `AMBIGUOUS_ROOT`
- `INVALID_JSON`
- `SCHEMA_INVALID`
- `REFERENCE_INVALID`
- `MANIFEST_MISMATCH`

Overall status:

- `COMPLETE` if all 7 are `INVENTORIED`
- `INCOMPLETE_WITH_BLOCKERS` otherwise

## Current outcome

All seven datasets are currently **MISSING** from this host.

resolved datasets: 0
inventoried datasets: 0
blocked datasets: 7
inventory_status: INCOMPLETE_WITH_BLOCKERS

This is a valid Step 91 outcome. The infrastructure of the
inventory is complete and correct. It is ready to run the moment
the datasets arrive.

## Next gate

If the seven datasets arrive complete and valid → **Step 92:
tokenizer-based sizing with the local BGE-M3 tokenizer** (no model
weights, no embeddings).

Otherwise → **Step 92: dataset remediation and envelope freeze**
before any tokenizer step.