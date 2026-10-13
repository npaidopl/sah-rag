# Artifact Provenance and Sidecar Manifests

## Status

The project implements a reusable sidecar-manifest layer for generated RAG artifacts.

Current manifest version:

```text
1.0
```

## Purpose

Every generated artifact must be independently verifiable and traceable to:
- its producing component,
- the exact component configuration,
- its consumed input artifacts,
- its source-code revision,
- its execution profile,
- its dataset,
- its record count,
- its byte size,
- its SHA-256 fingerprint.

## Sidecar Convention

A generated JSONL artifact is accompanied by a JSON sidecar.
Example:
- `parsed_documents.jsonl` -> `parsed_documents.manifest.json`
- `child_chunks.jsonl` -> `child_chunks.manifest.json`

The sidecar does not contain the artifact records. It contains identity, provenance, integrity, and production metadata.

## Artifact Integrity

Every sidecar records:
- `artifact_type`
- `artifact_path`
- `artifact_sha256`
- `artifact_size_bytes`
- `record_count`
- `created_at`
- `completed`

The SHA-256 fingerprint is calculated through streaming file reads. Large artifacts are not loaded fully into memory.

## Component Provenance

Every artifact records the component that produced it:
- component type,
- component ID,
- component version,
- configuration hash,
- normalized component configuration.

Examples of component types include:
- parser,
- chunker,
- tokenizer,
- embedder,
- index builder,
- retriever,
- evaluator.

## Input Provenance

A downstream artifact may reference one or more consumed artifacts.
Every input reference records:
- input artifact type,
- absolute input path,
- SHA-256 fingerprint,
- byte size,
- optional record count.

For the initial parsing and chunking flow (`parsed_documents.jsonl` -> `child_chunks.jsonl`), the child-chunk manifest records the exact SHA-256 fingerprint of the parsed-document artifact. This prevents a chunk artifact from being incorrectly associated with a different parsing run.

## Provenance Chain

The intended provenance chain is:
`canonical manifest` -> `parsed-document manifest` -> `child-chunk manifest` -> `embedding-shard manifest` -> `vector-store collection manifest` -> `prediction manifest` -> `evaluation manifest`

Every downstream stage must reference the exact fingerprint of its upstream artifacts.

## Atomic Publication

Artifact data and manifests are written atomically. The implementation uses temporary files and atomic replacement. An incomplete temporary artifact is removed if processing fails.

A completed sidecar is published only after:
- the artifact exists,
- artifact production has completed,
- record counting has completed,
- the SHA-256 fingerprint has been calculated,
- the manifest has passed typed validation.

## Code Revision

The sidecar may record the current Git commit hash. When the repository has uncommitted changes, the commit hash alone does not fully represent the working source tree.

For final benchmark execution:
- the working tree must be clean,
- all relevant implementation changes must be committed,
- generated manifests must record that committed revision.

Development artifacts created from a dirty working tree are diagnostic artifacts and must not be treated as final benchmark outputs.

## Execution Profile

The sidecar may record the execution profile, such as:
- `local_cpu`
- `colab_gpu`
- `local_cuda`
- `precomputed_import`

The execution profile is separate from the scientific component configuration. Equivalent scientific configurations may be executed in different environments while preserving the same parser or chunker configuration fingerprint.

## Record Count

For JSONL artifacts, record count means the number of non-empty JSONL records.
Examples:
- parsed artifact record count equals the number of parsed documents,
- child artifact record count equals the number of child chunks,
- prediction artifact record count equals the number of evaluated queries.

## Sample Validation

The initial sample provenance chain produced:
- Parsed documents: 5
- Child chunks: 8

Validation confirmed:
- Parsed checksum valid: true
- Chunk checksum valid: true
- Chunk input matches parsed: true
- Input size matches parsed: true
- Input record count matches parsed: true
- Code revisions equal: true

## JSON Schema

The sidecar model is exported as `schemas/artifact_sidecar_manifest.schema.json`.
The JSON Schema and Python model are generated from the same Pydantic implementation.

## Security and Privacy

Sidecars must not contain:
- access tokens,
- API keys,
- passwords,
- private environment-variable values.

Artifact paths may reveal machine-specific directory structures. Before public release, manifests may require path normalization or conversion to repository-relative and artifact-root-relative paths.

## Non-Goals

The sidecar layer does not:
- store artifact contents,
- replace experiment manifests,
- replace canonical dataset manifests,
- guarantee semantic quality,
- repair corrupted artifacts,
- upload artifacts to remote storage.
