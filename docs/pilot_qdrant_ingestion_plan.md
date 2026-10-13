# Pilot Qdrant Local-Persistence Ingestion Plan

**Step:** 84
**Status:** PLANNED (planning only)
**Scope:** Infrastructure validation only. Not a publishable result.

## Purpose

Freeze the Qdrant collection identity, vector schema, point-identity
policy, payload contract, storage-root location, and acceptance gates
for the pilot synthetic dataset. This step does **not** create any
collection, does **not** ingest any vectors, and does **not** execute
retrieval.

## Backend

- Type: Qdrant
- Mode: local persistence
- Server required: false
- Docker required: false
- Network required: false
- Credentials required: false
- Client construction: `QdrantClient(path=...)` only.
  The client is **not** instantiated in Step 84.

## Storage root

outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system/qdrant

The shared production path `D:\rag_work\qdrant_storage` is not used.

## Collection

- Logical name: `infrastructure_pilot_rag_system_bge_m3_v1`
- Physical name builder: `rag_system.storage.qdrant.fingerprint_collection_name`
- Algorithm:
  `rag_{normalized_dataset}_{12hex}` where `12hex` is the first 12
  hexadecimal characters of `sha256(embedding_config_hash)`.

Current values (from the Step 83 embedding catalog):

- `embedding_config_hash`: `bc92fc9c345313720b46bed3ce5f5459d139204b42cbdbc5cd105a7589002d12`
- physical collection name: `rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0`

### Spec divergence (informational only)

The Step 84 spec originally proposed
`infrastructure_pilot_rag_system_bge_m3_v1_{16hex}`. The project-owned
helper `fingerprint_collection_name` already implements a deterministic
scheme and is the function that Step 85 will invoke. This plan freezes
the actual code behavior, not the speculative format.

## Vectors

### Dense

- Name: `dense`
- Dimension: 1024
- Distance: `COSINE`
- Datatype: `float32`

### Sparse

- Name: `sparse`
- Type: learned sparse
- Indices sorted ascending, unique, non-negative
- Values finite and non-negative

## Point identity

- Function: `rag_system.storage.qdrant.build_qdrant_point_id`
- Algorithm: `uuid5`
- Namespace: `URL`
- Input format: `rag-chunk:{chunk_id}`

## Payload contract

Frozen in `baseline_audit/pilot_qdrant_payload_contract.yaml`.
The payload builder `rag_system.storage.qdrant.build_payload` emits:

`chunk_id`, `parent_id`, `doc_id`, `dataset`, `chunk_index`,
`parent_chunk_index`, `text_normalized`, `normalized_char_start`,
`normalized_char_end`, `token_start`, `token_end`, `token_count`,
`model_id`, `model_revision`, `embedding_config_hash`,
`embedding_shard_id`.

### Spec divergences (informational only)

The spec originally requested `dataset_id`, `source_order`, and
`content_sha256`. The frozen payload builder does not emit those names
or those values. The actual field set is recorded here for
traceability. If Step 85 requires a payload-level content checksum, the
builder must be amended explicitly and this contract bumped to
version 0.2.0.

## Acceptance gates

- embedding manifest SUCCEEDED
- embedding checksums valid
- record count exact (12)
- point count exact (12)
- point IDs unique and deterministic
- dense vectors valid (dim 1024, L2-normalized)
- sparse vectors valid (ascending unique non-negative indices)
- payloads complete
- lineage complete (`point_id -> chunk_id -> parent_id -> doc_id`)

## Downstream state

| Action | Enabled |
|---|---|
| environment inspection | true |
| ingestion planning | true |
| collection creation | false |
| vector ingestion | false |
| retrieval | false |
| evaluation | false |
| generation | false |

## Next gate

**Step 85: Bounded local-persistence Qdrant ingestion of exactly 12 points.**

No collection is created and no vector is ingested until Step 85.