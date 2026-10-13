# Pilot Parser and Chunker Execution

**Status:** Step 81 bounded execution
**Source:** `processed/canonical/infrastructure-pilot-synthetic-v1/`
**Output:** `outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system/`

## Scope

Runs the real `rag_system` parser and chunker end to end on the pilot
canonical envelope. No embedding, index, retrieval, evaluation, or
generation. No network. No foreign code.

## Pipeline
12 canonical documents
-> DeterministicStructuralParser (v0.3.0)
-> semantic parents
-> HierarchicalChildChunker (v0.3.0)
with RegexOffsetTokenizer, chunk_size=256, overlap=32
-> child chunks

## Source reference chunks

The 24 Step 78 chunks are **validation reference only**:

- not copied as authoritative chunks
- no byte equality, boundary equality, or count equality required
- used only to confirm that both source and derived artifacts cover
  the same 12 canonical documents and the same 6 content groups

## Output
outputs/infrastructure_pilot/infrastructure-pilot-synthetic-v1/rag_system/
parsing/
parsed_documents.jsonl
parsing.manifest.json
checksums.json
chunking/
child_chunks.jsonl
chunking.manifest.json
checksums.json

`outputs/` is git-ignored.

## Invariants

- 12 parsed documents (one per canonical doc)
- every parsed document has at least one child chunk
- chunk IDs unique
- parent IDs reference real semantic parents
- duplicate-content pairs stay distinct (`doc-a` != `doc-b`)
- no downstream artifacts (no embeddings, no retrieval output)

## Downstream approvals

Still `false`:

- embedding
- index
- retrieval
- evaluation
- generation

## Next gate

Step 82 — Embedding Execution Plan: BGE-M3 revision pin, license
review, cache identity, batch size, CPU benchmark protocol, Colab
transfer protocol, and Qdrant collection identity. Only after that do
we size local-vs-Colab embedding time for the 7 real datasets.
