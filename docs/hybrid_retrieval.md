# Hybrid Retrieval and Parent Aggregation

## Status

The project implements real BGE-M3 query embedding and hybrid retrieval over Qdrant.

Supported retrieval modes are:
- dense retrieval,
- learned-sparse retrieval,
- Reciprocal Rank Fusion,
- parent-level aggregation.

## Query Embedding

Queries are encoded with the same pinned BGE-M3 model revision used for document chunks.

The query encoder produces:
- one normalized 1024-dimensional dense vector,
- one learned-sparse lexical vector.

The query configuration must match the indexed chunk configuration.

## Dense Retrieval

Dense retrieval uses the named Qdrant vector:
```text
dense
Configuration:

dimension: 1024
distance: cosine
Dense retrieval captures semantic similarity.

Learned-Sparse Retrieval
Learned-sparse retrieval uses the named vector:
Plaintext

sparse
Sparse retrieval captures lexical and token-level relevance through BGE-M3 lexical weights. Its scores are not directly comparable to cosine scores.

Reciprocal Rank Fusion
Dense and sparse rankings are fused through Qdrant's native Reciprocal Rank Fusion query. RRF combines result ranks rather than directly adding heterogeneous scores.
The retrieval flow is:
Plaintext

dense prefetch + sparse prefetch -> RRF fusion -> ranked child chunks
The candidate limit must be at least as large as the final result limit.

Parent-Level Aggregation
Retrieved child chunks are grouped by parent_id. The parent score formula is:
Plaintext

parent score = best child score + 0.05 × sum(other child scores)
This policy:

preserves the dominance of the strongest child,
grants a small evidence bonus to parents with multiple matches,
limits bias toward large parents.
Initial Real Query Test
The initial query was:
Plaintext

electrical appliance casing and supporting plate
The query produced nine non-zero learned-sparse features. Dense, sparse, and hybrid RRF retrieval all ranked the same relevant chunk first:
Plaintext

chunk-clerk-000000-06ea0646fbf2
Query encoding on CPU required approximately 2.7 seconds.
