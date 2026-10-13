# Pilot Canonical Adapter

**Status:** Step 80
**Source:** `data/pilot/infrastructure-pilot-synthetic-v1/`
**Target:** `processed/canonical/infrastructure-pilot-synthetic-v1/`
**Adapter:** `pilot_to_rag_system_canonical` v0.1.0

## Purpose

Convert the immutable Step 78 synthetic pilot envelope into the shape
that `rag_system` reads, without executing anything and without
modifying the source.

## Source envelope (Step 78)

data/pilot/infrastructure-pilot-synthetic-v1/
documents.jsonl 12
chunks.jsonl 24 (validation reference only)
queries.jsonl 6
judgments.jsonl 12
dataset.manifest.json
checksums.json
## Target envelope (Step 80)
processed/canonical/infrastructure-pilot-synthetic-v1/
documents.jsonl 12 CanonicalDocument
examples.jsonl 6 CanonicalExample
validation.json 1 PilotAdapterValidation
manifest.json 1 PilotAdapterManifest
adapter.mapping.json 1 mapping
checksums.json 5 file checksums

## Mapping

| Source | Target | Rule |
|---|---|---|
| documents.jsonl | documents.jsonl | doc_id, text, checksum preserved |
| queries.jsonl + judgments.jsonl | examples.jsonl | one CanonicalExample per query, ground truth = 2 relevant doc_ids |
| chunks.jsonl | (not copied) | validation reference only; rag_system chunker regenerates |
| dataset.manifest.json | manifest.json | re-shaped |
| checksums.json | (absorbed) | re-emitted per target file |

## What is preserved

- 12 source documents → 12 target documents, same `doc_id`
- 6 duplicate-content pairs preserved (`source_content_group_id`)
- Exact `text_normalized`
- Source `source_text_sha256` recorded in target metadata
- Source manifest SHA recorded in every target record

## What is NOT created

- Reference answers
- Generated answers
- Answer text
- Any LLM completion

## Source chunks are not authoritative

The 24 source chunks exist only as a **validation reference**:

- The adapter checks that they reconstruct their parent documents.
- The target envelope does not include them as `ChildChunk` records.
- The real `rag_system` chunker regenerates chunks during execution,
  using `HierarchicalChildChunker` (256 tokens, 32 overlap).

This prevents bypassing the architecture's own chunking step.

## Safety

Nothing runs in the adapter:
parser_executed: false
chunker_executed: false
embedding_executed: false
retrieval_executed: false
evaluation_executed: false
generation_executed: false
network_required: false

## Determinism

Two independent materializations produce byte-identical files. The
adapter is pure: same input → same output, no timestamps in artifacts.

## Lineage

Every target document and example carries:

- `source_dataset_id`
- `source_dataset_revision`
- `source_manifest_sha256`
- `source_doc_id` / `source_query_id`
- `source_content_group_id`
- `adapter_id`, `adapter_version`

## Immutability

- The source envelope is never modified. Verified by
  `test_source_artifacts_remain_unchanged`.
- The target requires `--overwrite` to replace an existing directory.
- Any semantic change to the adapter requires a new `adapter_version`.

## Next gate

After Step 80 the pipeline can be pointed at the target envelope. But:

parser_execution_approved: false
chunker_execution_approved: false
embedding_execution_approved: false
retrieval_execution_approved: false

Only the *envelope* is ready. Execution approval is Step 81+.
