# Embedding Shard Artifacts

## Status

The project implements typed, checksummed embedding-shard artifacts.

The current infrastructure supports:

- dense embeddings,
- learned-sparse embeddings,
- hybrid dense and learned-sparse embeddings,
- deterministic shard identities,
- source-chunk membership validation,
- atomic shard publication,
- sidecar provenance manifests.

No real BGE-M3 embeddings are produced at this stage.

## Directory Structure

Each embedding shard uses the following structure:

- `header.json`
- `records.jsonl`
- `shard.manifest.json`

## Header

The shard header records configuration shared by all records:

- shard ID
- shard index
- dataset
- model ID
- exact model revision
- tokenizer ID
- embedding configuration hash
- embedding kind
- dense dimension
- dense precision
- dense normalization policy
- maximum records

## Embedding Records

Each `ChunkEmbeddingRecord` preserves:

- `chunk_id`
- `parent_id`
- `doc_id`
- `dataset`
- `model ID`
- `model revision`
- `tokenizer ID`
- `embedding configuration hash`
- optional dense vector
- optional learned-sparse vector
- dense-vector metadata
- source chunk fingerprint

At least one embedding representation is required.

## Dense Representation

Dense vectors record:

- exact vector dimension,
- storage precision,
- normalization status,
- finite numeric values.

A dense vector must have exactly the declared dimension.

## Learned-Sparse Representation

Learned-sparse vectors use aligned arrays:

- `indices`
- `values`

Sparse indices must be:

- non-negative,
- unique,
- sorted,
- aligned with the values array.

Sparse values must be finite.

## Hybrid Representation

A hybrid shard contains both:

- one dense vector per chunk,
- one learned-sparse vector per chunk.

The shard writer validates that every record has the representations required by the shard header.

## Source Membership

Every embedding record must reference a `chunk_id` that exists in the source child-chunk artifact.

The writer rejects:

- unknown source chunk IDs,
- duplicate embedding chunk IDs,
- duplicate source chunk IDs,
- record and header configuration mismatches.

A shard may contain a subset of the source chunks.

The sidecar therefore records:

- source chunk count,
- embedded chunk count,
- whether the shard provides complete source coverage.

## Shard Identity

Shard IDs are deterministic functions of:

- dataset ID,
- model ID,
- embedding configuration hash,
- shard index.

Equivalent inputs produce equivalent shard IDs.

## Provenance

The shard sidecar references:

- the source child-chunk artifact,
- the embedding shard header.

For each input, the sidecar records:

- artifact type,
- absolute path,
- SHA-256 fingerprint,
- byte size,
- record count.

The sidecar additionally records:

- shard identity,
- component configuration,
- code revision,
- execution profile,
- experiment ID when available.

## Atomic Publication

Shard publication is fail-safe.

If any stage fails, the writer removes:

- `header.json`
- `records.jsonl`
- `shard.manifest.json`

A partially completed shard is not treated as a published artifact.

## Synthetic Embeddings

The repository includes deterministic synthetic embeddings for infrastructure tests.

These vectors:

- are derived from chunk IDs,
- are deterministic,
- have normalized dense components,
- contain synthetic sparse values,
- do not encode semantic meaning.

Synthetic embedding artifacts must include:

```json
{
  "synthetic": true,
  "semantic_embeddings": false
}
```

They must never be used for retrieval-quality evaluation.

## Initial Synthetic Validation

The initial infrastructure test produced:

- Source chunks: 8
- Embedding records: 8
- Unique embedded chunks: 8
- Complete source coverage: true
- Records checksum valid: true

The shard was explicitly identified as synthetic and non-semantic.

## Planned Production Configuration

The initial production model family is:

- `BAAI/bge-m3`

Before producing real embeddings, the pipeline must record:

- pinned model revision,
- tokenizer revision,
- PyTorch version,
- FlagEmbedding version,
- device,
- precision,
- batch size,
- dense dimension,
- normalization policy,
- learned-sparse configuration.

## JSON Schemas

The embedding artifact schemas are exported as:

- `schemas/chunk_embedding.schema.json`
- `schemas/embedding_shard_header.schema.json`

They are generated from the same Pydantic models used by the runtime.

## Non-Goals

The shard infrastructure does not yet:

- load BGE-M3 model weights,
- generate real semantic embeddings,
- write Qdrant collections,
- perform retrieval,
- select optimal batch size,
- select optimal storage precision.