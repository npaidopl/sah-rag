"""Adapters for externally defined retrieval architectures.

This package exposes exactly two comparator families used in the
SAH-RAG project:

* LongRAG: a controlled adaptation of the LongRAG paper, grouped
  adjacent documents into long retrieval units retrieved by dense
  similarity.
* Metadata-RAG: a paper-derived reimplementation of a metadata-driven
  multistage retrieval architecture, frozen under a clean-room
  specification.

Both families are executed under the shared evaluation protocol
described in ``docs/fair_benchmark_methodology.md``.

Rejected baselines (HippoRAG 2, RAPTOR, LightRAG, PathRAG, KAG,
MemoRAG, DisRetrieval, HiKEY) are documented in ``docs/`` and their
contracts are preserved in ``baseline_audit/``, but their code is
not part of this repository. See ``docs/remaining_baselines_sequencing.md``.
"""

from baselines.longrag import (
    ExpandedLongRAGDocument,
    LongRAGSourceSegment,
    LongUnit,
    LongUnitConstructionPolicy,
    LongUnitConstructionSummary,
    LongUnitTokenCounter,
    RankedLongUnit,
    build_long_unit_id,
    construct_adjacent_long_units,
    construct_long_units,
    construct_parent_grouped_long_units,
    expand_ranked_long_units_to_documents,
    summarize_long_units,
)
from baselines.longrag_bge_m3 import (
    BGE_M3_DENSE_DIMENSION,
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    BgeM3DenseEncoder,
)
from baselines.longrag_embeddings import (
    LongUnitEmbeddingCatalog,
    LongUnitEmbeddingRecord,
    LongUnitEmbeddingShardHeader,
)
from baselines.longrag_pipeline import (
    LongRAGEvaluationCase,
    ProjectedQuery,
    build_embedding_record,
    build_evaluation_case,
    build_source_segments,
    extract_relevant_ids,
    project_queries,
)
from baselines.longrag_reference_dense_index import (
    ReferenceDenseIndex,
)
from baselines.longrag_retrieval import (
    RetrievedLongUnit,
    retrieve_long_units,
    retrieved_to_ranked_long_units,
)
from baselines.longrag_storage import (
    LongRAGCollectionManifest,
    build_longrag_payload,
    build_longrag_point,
    build_longrag_point_id,
    create_longrag_collection,
    fingerprint_longrag_collection_name,
    ingest_longrag_records,
)

from baselines.metadata_rag_adapter import (
    MetadataRAGIndexedDocument,
    MetadataRAGProjectedQuery,
    build_indexed_document,
    build_indexed_documents,
    project_queries,
)
from baselines.metadata_rag_embedder import (
    FROZEN_DENSE_DIMENSION,
    FROZEN_MODEL_ID,
    FROZEN_MODEL_REVISION,
    BgeM3DenseEncoder as MetadataRAGBgeM3DenseEncoder,
    DenseEmbedderProtocol,
    EncoderConfig,
    MetadataRAGDocumentEmbedding,
    MetadataRAGQueryEmbedding,
    build_document_embedding,
    build_query_embedding,
)
from baselines.metadata_rag_evaluation_case import (
    MetadataRAGEvaluationCase,
    build_evaluation_case as build_metadata_rag_evaluation_case,
)
from baselines.metadata_rag_expansion import (
    ExpandedDocument,
    expand_ranked_items_to_documents,
)
from baselines.metadata_rag_metadata import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    MetadataConstraint,
    MetadataRecord,
    build_document_metadata,
    build_query_metadata,
    derive_query_constraints,
)
from baselines.metadata_rag_persistence import (
    read_state,
    write_state,
)
from baselines.metadata_rag_reference_index import (
    DenseIndexEntry,
    DenseSearchHit,
    ReferenceDenseIndex as MetadataRAGReferenceDenseIndex,
    build_reference_dense_index,
)
from baselines.metadata_rag_retrieval import (
    QueryRetrievalResult,
    RankedItem,
    RetrievalConfig,
    retrieve_batch,
    retrieve_query,
)
from baselines.metadata_rag_state import (
    MetadataRAGState,
    build_state,
)
from baselines.metadata_rag_windowing import (
    BgeM3WindowSplitter,
    SingleWindowSplitter,
    WindowSpan,
    WindowSplitterProtocol,
    build_bge_m3_window_splitter,
)

__all__ = [
    # LongRAG
    "ExpandedLongRAGDocument",
    "LongRAGSourceSegment",
    "LongUnit",
    "LongUnitConstructionPolicy",
    "LongUnitConstructionSummary",
    "LongUnitTokenCounter",
    "RankedLongUnit",
    "build_long_unit_id",
    "construct_adjacent_long_units",
    "construct_long_units",
    "construct_parent_grouped_long_units",
    "expand_ranked_long_units_to_documents",
    "summarize_long_units",
    "BGE_M3_DENSE_DIMENSION",
    "BGE_M3_MODEL_ID",
    "BGE_M3_MODEL_REVISION",
    "BgeM3DenseEncoder",
    "LongUnitEmbeddingCatalog",
    "LongUnitEmbeddingRecord",
    "LongUnitEmbeddingShardHeader",
    "LongRAGEvaluationCase",
    "ProjectedQuery",
    "build_embedding_record",
    "build_evaluation_case",
    "build_source_segments",
    "extract_relevant_ids",
    "project_queries",
    "ReferenceDenseIndex",
    "RetrievedLongUnit",
    "retrieve_long_units",
    "retrieved_to_ranked_long_units",
    "LongRAGCollectionManifest",
    "build_longrag_payload",
    "build_longrag_point",
    "build_longrag_point_id",
    "create_longrag_collection",
    "fingerprint_longrag_collection_name",
    "ingest_longrag_records",
    # Metadata-RAG
    "MetadataRAGIndexedDocument",
    "MetadataRAGProjectedQuery",
    "build_indexed_document",
    "build_indexed_documents",
    "FROZEN_DENSE_DIMENSION",
    "FROZEN_MODEL_ID",
    "FROZEN_MODEL_REVISION",
    "MetadataRAGBgeM3DenseEncoder",
    "DenseEmbedderProtocol",
    "EncoderConfig",
    "MetadataRAGDocumentEmbedding",
    "MetadataRAGQueryEmbedding",
    "build_document_embedding",
    "build_query_embedding",
    "MetadataRAGEvaluationCase",
    "build_metadata_rag_evaluation_case",
    "ExpandedDocument",
    "expand_ranked_items_to_documents",
    "EXTRACTOR_VERSION",
    "SCHEMA_VERSION",
    "MetadataConstraint",
    "MetadataRecord",
    "build_document_metadata",
    "build_query_metadata",
    "derive_query_constraints",
    "read_state",
    "write_state",
    "DenseIndexEntry",
    "DenseSearchHit",
    "MetadataRAGReferenceDenseIndex",
    "build_reference_dense_index",
    "QueryRetrievalResult",
    "RankedItem",
    "RetrievalConfig",
    "retrieve_batch",
    "retrieve_query",
    "MetadataRAGState",
    "build_state",
    "BgeM3WindowSplitter",
    "SingleWindowSplitter",
    "WindowSpan",
    "WindowSplitterProtocol",
    "build_bge_m3_window_splitter",
]