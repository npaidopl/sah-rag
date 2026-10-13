# Proposed RAG Architecture Specification

## Status

This document is an initial architecture contract.

It is not a final implementation specification. Architectural decisions may be revised through implementation, validation, ablation studies, and benchmark results.

## 1. Primary Objective

The proposed system is a structure-aware, hierarchical, hybrid-retrieval RAG architecture for heterogeneous long-context datasets.

The architecture separates:

1. canonical source documents,
2. parsed semantic parent entities,
3. indexed child retrieval chunks.

The primary architectural hypothesis is that retrieving small, focused child units while resolving them to larger semantic parent units can improve evidence retrieval and context reconstruction across heterogeneous long-context benchmarks.

## 2. Canonical Input Contract

The architecture consumes immutable canonical dataset artifacts.

For each dataset, the expected directory is:

processed/canonical/<dataset>/

Required files:

- documents.jsonl
- examples.jsonl
- validation.json
- manifest.json

The pipeline must not modify the canonical files.

The pipeline must not regenerate canonical document IDs or query IDs.

The pipeline must preserve:

- doc_id
- query_id
- dataset
- source_path
- source_row
- text_raw
- text_normalized
- metadata
- query.doc_ids
- ground_truth
- supplied_context
- options

## 3. Canonical Text Policy

Use text_normalized for:

- parser input,
- retrieval-oriented preprocessing,
- chunking,
- embeddings,
- indexing.

Preserve text_raw for:

- exact quotation,
- legal citation extraction,
- raw source validation,
- raw character-offset evaluation.

When derived text units are produced from text_normalized, retain mappings to:

- canonical doc_id,
- normalized character offsets,
- raw character offsets when an explicit mapping is available.

## 4. Offline Pipeline

The initial offline pipeline is:

1. canonical input validation,
2. optional derived preprocessing,
3. structural signal extraction,
4. semantic and structural parsing,
5. semantic parent construction,
6. hierarchical child chunking,
7. dense and learned-sparse embedding generation,
8. embedding artifact generation,
9. vector-store ingestion,
10. parent and canonical-document storage.

## 5. Parsing Strategy

The proposed parser is a layered hybrid parser.

Potential parsing layers include:

1. generic structural signal extraction,
2. deterministic hierarchy detection,
3. optional domain-aware structural rules,
4. statistical boundary classification,
5. embedding-assisted semantic boundary detection,
6. boundary reconciliation,
7. hierarchy construction,
8. fallback segmentation.

The parser must not assume that all documents are legal documents, books, contracts, or academic articles.

The parser must support documents without explicit structural markers.

The parser must record how every predicted boundary was produced.

Possible boundary provenance values include:

- explicit_heading
- numbered_hierarchy
- paragraph_break
- structural_rule
- semantic_drop
- statistical_classifier
- maximum_length_fallback
- document_end

## 6. Hierarchical Representation

The system uses three identity levels:

1. canonical source document,
2. parsed semantic parent,
3. child retrieval chunk.

Required identifiers:

- doc_id
- parent_id
- chunk_id

Every child chunk must resolve to exactly one semantic parent.

Every semantic parent must resolve to exactly one canonical document.

Future versions may support multiple hierarchy levels above the immediate semantic parent.

## 7. Child Chunking

Child chunks are produced inside semantic parents.

Initial candidate chunk sizes:

- 128 tokens
- 256 tokens
- 512 tokens

A 1024-token configuration may be used as a long-chunk control condition.

Chunk size is an experimental parameter and is not assumed to have a universally optimal value.

Every chunk must preserve:

- chunk_id
- doc_id
- parent_id
- chunk_index
- chunk text
- token count
- normalized character offsets
- raw character offsets when available
- chunker configuration
- parser configuration reference

## 8. Embedding Model

The initial embedding model is:

BAAI/bge-m3

Initial representations:

- dense embeddings
- learned sparse representations

Multi-vector representations are not required in the first implementation but may be evaluated later.

The embedding implementation must support:

- local CPU execution,
- CUDA execution,
- Colab GPU execution,
- artifact import from precomputed embedding shards.

Embedding artifacts must record:

- model name
- model revision
- tokenizer revision
- representation type
- vector dimension
- normalization
- maximum input length
- precision
- batch size
- execution device
- configuration hash

## 9. Storage

The initial storage design is:

- Qdrant for indexed child chunks,
- a parent/document store for semantic parents and canonical document references.

The parent/document store must be abstracted behind an interface.

Candidate implementations:

- SQLite
- DuckDB
- a second Qdrant collection

The first implementation should not assume that the parent store must be Qdrant.

## 10. Child Vector Collection

The child collection stores:

- dense vectors,
- learned sparse vectors,
- chunk_id,
- parent_id,
- doc_id,
- dataset,
- chunk text or chunk-text reference,
- relevant metadata,
- parser version,
- chunker version,
- embedder version.

## 11. Online Pipeline

The initial online pipeline is:

1. read canonical benchmark example,
2. preserve query_id,
3. apply the allowed query document scope,
4. generate dense query representation,
5. generate learned-sparse query representation,
6. perform dense retrieval,
7. perform sparse retrieval,
8. fuse rankings,
9. return top-k child chunks,
10. resolve semantic parents,
11. resolve canonical source documents,
12. produce structured retrieval output,
13. route the output to dataset-specific evaluation.

## 12. Rank Fusion

The first fusion method is Reciprocal Rank Fusion.

RRF is treated as a rank-fusion method, not as a neural reranker.

Alternative fusion methods may include:

- DBSF,
- weighted score fusion.

A cross-encoder or another neural reranker may be added after rank fusion as a separate stage.

## 13. Retrieval Scope

When query.doc_ids defines the valid or intended document scope, the pipeline must support restricting retrieval to those documents.

Restricted-scope and unrestricted-corpus experiments must not be mixed without explicit labeling.

## 14. Retrieval Output Contract

Every prediction record should include:

- experiment_id
- architecture_id
- dataset
- query_id
- referenced doc_ids
- retrieved chunk IDs
- retrieved parent IDs
- retrieved canonical doc_ids
- dense ranks and scores
- sparse ranks and scores
- fused ranks and scores
- final ranking
- resolved context
- retrieval configuration
- parser configuration
- chunking configuration
- embedding configuration
- latency and resource measurements when enabled

## 15. Parsing Evaluation

Candidate parsing metrics include:

- exact boundary precision
- exact boundary recall
- exact boundary F1
- tolerant boundary precision
- tolerant boundary recall
- tolerant boundary F1
- Pk
- WindowDiff
- segment Intersection over Union
- over-segmentation rate
- under-segmentation rate
- parent-assignment accuracy
- hierarchy-level accuracy
- downstream retrieval utility

A manually annotated parsing subset may be created if the canonical benchmarks do not contain gold structural boundaries.

## 16. Retrieval Evaluation

Candidate retrieval metrics include:

- Recall@K
- Precision@K
- Hit Rate@K
- MRR
- MAP
- nDCG@K
- child-level recall
- parent-level recall
- source-document recall
- evidence coverage
- retrieved token count
- context redundancy
- unique parent count

Evaluation must be routed according to ground_truth.ground_truth_type.

## 17. Experimental Parameters

Initial experimental parameters include:

- parser configuration
- chunk size
- chunk overlap
- top-k
- dense-only retrieval
- sparse-only retrieval
- hybrid retrieval
- fusion method
- parent-resolution strategy
- retrieval scope

Not all parameter combinations must be executed on all datasets.

## 18. Execution Environments

The codebase must remain execution-environment independent.

Supported execution profiles should include:

- local CPU
- local CUDA GPU
- Google Colab CUDA GPU
- precomputed embedding artifact import

The scientific pipeline must not depend specifically on Google Colab.

Google Colab is one possible execution environment for GPU-dependent stages.

## 19. Immutability and Artifact Policy

Canonical inputs are immutable.

All derived outputs must be written outside the canonical dataset directories.

Derived artifacts include:

- parsed documents
- semantic parents
- child chunks
- embeddings
- Qdrant snapshots
- predictions
- metrics
- logs
- manifests

Every derived artifact must be traceable to:

- canonical input fingerprint
- code revision
- configuration
- model revision
- experiment ID

## 20. Initial Non-Goals

The first implementation does not require:

- answer generation,
- cloud production deployment,
- a user interface,
- a universal perfect parser,
- multi-vector retrieval,
- a neural reranker,
- all baseline systems,
- optimization for every dataset.

The first goal is a correct end-to-end retrieval pipeline.