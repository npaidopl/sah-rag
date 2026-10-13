# Canonical Dataset Integration Contract

## Input Artifacts

Each dataset is expected under:

processed/canonical/<dataset>/

Required files:

- documents.jsonl
- examples.jsonl
- validation.json
- manifest.json

## Document Source

documents.jsonl is the only canonical document-ingestion source.

Each document must preserve:

- doc_id
- dataset
- text_raw
- text_normalized
- title
- source_path
- source_row
- metadata

## Benchmark Source

examples.jsonl is the only canonical benchmark-execution source.

Each example may include:

- query
- ground_truth
- supplied_context
- options
- example-level metadata

## Identity Rules

The pipeline must preserve doc_id and query_id unchanged.

Derived identifiers must never replace canonical identifiers.

Derived IDs must maintain explicit references to canonical IDs.

## Streaming Rule

JSONL files must be processed through streaming iteration.

The implementation must not use read_text().splitlines() for full canonical files.

## Text Policy

Use text_normalized for parsing, chunking, embeddings, and retrieval.

Preserve text_raw for exact quotations, legal citations, source validation, and raw-offset evaluation.

## Scope Policy

query.doc_ids defines the allowed or intended document scope when required by the benchmark protocol.

The pipeline must not mix contexts across isolated per-example benchmarks unless the experiment explicitly enables unrestricted retrieval.

## Output Policy

No derived output may be written inside processed/canonical/<dataset>/.

Derived outputs must use experiment-specific output directories.

## Ground-Truth Policy

Evaluation must be selected according to ground_truth.ground_truth_type.

No-answer examples must be preserved.

Multiple accepted answers must remain separate.

Multiple-choice labels and answer text must both be preserved.

## Validation Policy

Before ingestion, the pipeline must verify:

- required files exist,
- JSONL records are readable,
- required identifiers exist,
- doc_id values are unique within the dataset,
- query_id values are unique within the dataset,
- referenced doc_ids can be resolved where required,
- validation.json reports successful canonical validation,
- manifest.json is structurally valid.