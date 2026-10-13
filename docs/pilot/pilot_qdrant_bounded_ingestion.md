# Pilot Qdrant Bounded Local-Persistence Ingestion

**Step:** 85
**Status:** APPROVED_FOR_BOUNDED_EXECUTION
**Scope:** Infrastructure validation only. Not a publishable result.

## Purpose

Perform the first bounded local-persistence Qdrant ingestion of
exactly 12 deterministic points for the pilot synthetic dataset.

The following stages remain disabled:

- retrieval execution
- evaluation execution
- generation execution
- network access
- server mode
- docker execution

## Inputs

- Embedding root:
  `outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system/embedding`
- Records: 12 (validated in Step 83)
- Chunks: 12 (`child_chunks.jsonl`, matched by `chunk_id`)
- Model: `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`

## Backend

- Type: Qdrant
- Mode: local persistence (embedded)
- Client initialization: `QdrantClient(path=...)`
- Server, docker, network, credentials: all required = false

## Storage layout
outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system/
├── embedding/ (Step 83; untouched by Step 85)
├── qdrant/ (Qdrant local persistence files)
└── qdrant_ingestion/ (audit artifacts)
├── collection.manifest.json
├── point_inventory.jsonl
├── payload_inventory.jsonl
└── checksums.json

## Collection

- Logical name: `infrastructure_pilot_rag_system_bge_m3_v1`
- Physical name: `rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0`
- Builder: `rag_system.storage.qdrant.fingerprint_collection_name`
- Expected point count: 12

## Vectors

- Dense: `dense`, 1024, COSINE
- Sparse: `sparse`, learned lexical weights

## Point identity

- Builder: `rag_system.storage.qdrant.build_qdrant_point_id`
- Algorithm: uuid5, namespace URL, input `rag-chunk:{chunk_id}`

## Payload

Builder: `rag_system.storage.qdrant.build_payload`
Fields frozen in `pilot_qdrant_payload_contract.yaml`.

## Execution sequence

1. Validate Step 84 plan and Step 83 embedding outputs.
2. Refuse to run if storage root exists and `--overwrite` is not given.
3. Open client 1 (`QdrantClient(path=...)`).
4. `create_hybrid_collection(...)`.
5. `ingest_embedding_records(..., batch_size=12)` in one batch.
6. Close client 1.
7. Open client 2 on the same path.
8. Verify collection existence, vector schema, exact point count 12.
9. Scroll points (metadata only, no query vector).
10. Compare persisted point IDs with the Step 84 deterministic plan.
11. Validate payloads (all required fields present, no prohibited field).
12. Close client 2.
13. Write audit inventory, collection manifest, checksums.
14. Update audit-side execution manifest.

Retrieval, evaluation and generation remain disabled. The point
scroll for round-trip validation is metadata inspection only; no
query vector is ever submitted.

## Client lock policy

Only one local client is open at a time. The runner never overlaps
client 1 and client 2 on the same storage path.

## Storage safety

- The storage root must be inside
  `outputs/infrastructure_pilot/...`
- `--overwrite` is rejected if the audit root is outside the pilot tree.
- Recursive delete is applied only to the exact Qdrant storage root,
  never to a parent directory.

## Acceptance gates

- embedding manifest SUCCEEDED
- embedding checksums valid
- collection exists with exact physical name
- dense and sparse schemas valid
- persisted point count 12
- persisted point IDs match the Step 84 plan
- payloads complete, no prohibited fields
- client restart validates persisted collection
- storage size within 100 MiB
- retrieval, evaluation, generation all false

## Next gate

**Step 86: Bounded hybrid retrieval plan for six pilot queries (planning only).**