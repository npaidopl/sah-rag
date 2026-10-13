# Hierarchical Child Chunker

## Status

The hierarchical child chunker is implemented as a deterministic, parent-aware chunking component.

Current version:

```text
0.2.0
```

The current implementation uses a dependency-free regex tokenizer for testing and integration validation.
The final benchmark implementation will use an offset-capable tokenizer compatible with the selected embedding model.

## Purpose

The chunker converts validated ParsedDocument artifacts into indexed ChildChunk artifacts.
The parser and chunker have different responsibilities:

* the parser identifies or preserves semantic parent units,
* the chunker creates retrieval-sized child units inside each parent.

A child chunk is never allowed to cross a semantic-parent boundary.

## Input

The chunker consumes:

```text
ParsedDocument
└── SemanticParent
```

Each semantic parent provides:

* canonical document identity,
* semantic-parent identity,
* normalized text,
* normalized character offsets,
* raw text and offsets when available,
* parser identity and configuration fingerprint.

## Output

The chunker produces ChildChunk records.
Each child chunk preserves:

* chunk_id
* parent_id
* doc_id
* dataset
* global chunk index
* parent-local chunk index
* normalized text
* raw text when available
* normalized character span
* parent character span
* raw character span when available
* parent-local token span
* token count
* parser identity
* chunker identity
* tokenizer identity
* chunking configuration fingerprint

## Parent Boundary Policy

The chunker enforces:

```text
allow_cross_parent_chunks = false
```

Every chunk belongs to exactly one semantic parent.
Every semantic parent belongs to exactly one canonical document.
The resolution chain is:

```text
chunk_id → parent_id → doc_id
```

## Window Policy

The current default configuration is:

```text
chunk_size = 256
chunk_overlap = 32
minimum_final_chunk_tokens = 128
```

The effective minimum final chunk size defaults to half of the configured chunk size.
For each semantic parent:

* tokenize the parent with character offsets,
* create overlapping token windows,
* preserve the configured overlap for normal windows,
* anchor the final window to the end when the remaining tail would be too small,
* preserve complete source-token coverage.

## Short Parent Policy

If an entire parent contains fewer tokens than chunk_size, it remains one shorter child chunk.
The chunker does not:

* pad the chunk,
* repeat unrelated text,
* merge it with another semantic parent,
* discard it solely because it is short.

Short canonical records remain valid retrieval units.

## Tail Policy

When a parent is longer than chunk_size and the normal final window would contain fewer than the configured minimum number of tokens, the final window is shifted backward and anchored to the end of the parent.
This prevents tiny tail chunks.
The anchored final window may overlap more heavily with the previous window. This behavior is deterministic and configuration-dependent.
Future experimental policies may include:

* preserving the short tail,
* merging the tail with the previous chunk,
* replacing the previous window with an anchored final window.

## Deterministic Identity

Chunk identifiers are generated from:

* dataset ID,
* canonical document ID,
* semantic parent ID,
* chunker configuration fingerprint,
* parent-local chunk index.

Repeated chunking with the same:

* parsed document,
* tokenizer,
* chunker version,
* chunking configuration

produces identical chunk IDs.

## Tokenizer Abstraction

The chunker depends on the OffsetTokenizer interface.
Required behavior:

```text
text → ordered tokens → exact character offsets
```

The initial tokenizer is:

```text
regex-unicode-v1
```

This tokenizer exists for:

* unit tests,
* deterministic smoke tests,
* pipeline integration,
* offset validation.

It is not the final benchmark tokenizer.
A later adapter will use the tokenizer associated with:

```text
BAAI/bge-m3
```

The tokenizer identity is included in the chunker configuration fingerprint, so chunks produced by different tokenizers cannot silently share the same artifact identity.

## Streaming Pipeline

The parsed-to-chunk pipeline reads:

```text
parsed_documents.jsonl
```

through streaming iteration and writes:

```text
child_chunks.jsonl
```

through an atomic temporary file.
A completed output becomes visible only after successful processing.
If processing fails, the incomplete temporary artifact is removed.

## Validation

The implementation validates:

* unique chunk IDs,
* contiguous global chunk indexes,
* resolvable parent IDs,
* matching document IDs,
* matching dataset IDs,
* child spans contained inside parent spans,
* exact parent-source text mapping,
* token count consistent with token span,
* complete token coverage,
* configured overlap and tail behavior.

## Current Integration Result

A five-document clerk sample produced:

```text
Parsed documents: 5
Semantic parents: 5
Child chunks: 8
Parents without chunks: 0
Unique chunk IDs: 8
Tokenizer: regex-unicode-v1
Chunker version: 0.2.0
```

Observed token-count range:

```text
6–256
```

The six-token chunk belongs to a very short canonical source record and is therefore valid.

## Experimental Parameters

Planned child-size configurations include:

* 128 tokens,
* 256 tokens,
* 512 tokens,
* 1024-token control condition.

Planned top_k values are query-time parameters and do not require rebuilding embeddings.
Each chunk-size configuration requires separate:

* child artifacts,
* embeddings,
* vector-store collection,
* configuration fingerprint.

## Non-Goals

The current chunker does not:

* infer semantic boundaries,
* load BGE-M3,
* generate embeddings,
* perform retrieval,
* cross semantic-parent boundaries,
* change canonical inputs,
* repair text-quality problems,
* select the universally optimal chunk size.
