# Qdrant Hybrid Vector Storage

## Status

The project supports validated insertion of BGE-M3 hybrid embeddings into Qdrant.

The initial implementation has been verified with Qdrant local in-memory mode.

## Collection Schema

Every point contains two named vector representations:

- `dense`
- `sparse`

The dense vector configuration is:
- Size: 1024
- Distance: Cosine

The learned-sparse vector configuration uses dot-product scoring.

## Point Identity & Payload

Every point ID is a deterministic UUID derived from the stable `chunk_id`. The payload preserves all metadata, structural offsets, and provenance hashes.

## Initial Sample Result

Eight real BGE-M3 hybrid embedding records were inserted into an in-memory collection successfully, with valid dense and sparse self-matches.
