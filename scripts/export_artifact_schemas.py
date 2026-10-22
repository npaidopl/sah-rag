"""Export JSON schemas for canonical system artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from sah_rag.artifacts import (
    ArtifactSidecarManifest,
)
from baselines import (
    ExpandedHippoRAG2Document,
    ExpandedLightRAGDocument,
    ExpandedPathRAGDocument,
    ExpandedRAPTORDocument,
    HippoRAG2CanonicalMapping,
    HippoRAG2CanonicalSource,
    HippoRAG2ExtractionResult,
    HippoRAG2FactScore,
    HippoRAG2GraphCatalog,
    HippoRAG2GraphEdge,
    HippoRAG2GraphNode,
    HippoRAG2NerResult,
    HippoRAG2OpenIECatalog,
    HippoRAG2PPRResult,
    HippoRAG2PPRSeed,
    HippoRAG2RecognitionArtifact,
    HippoRAG2Triple,
    LightRAGCanonicalSource,
    LightRAGChunkMapping,
    LightRAGEntity,
    LightRAGExtractionCatalog,
    LightRAGExtractionProvenance,
    LightRAGQueryContextArtifact,
    LightRAGRankedChunk,
    LightRAGRelationship,
    LightRAGStructuredEvidence,
    LongRAGCollectionManifest,
    LongRAGSourceSegment,
    LongUnit,
    LongUnitConstructionSummary,
    LongUnitEmbeddingCatalog,
    LongUnitEmbeddingRecord,
    LongUnitEmbeddingShardHeader,
    PathRAGCandidatePath,
    PathRAGCanonicalSource,
    PathRAGComparatorCatalog,
    PathRAGGraphEdge,
    PathRAGGraphNode,
    PathRAGPathScore,
    PathRAGPropagationStep,
    PathRAGPropagationTrace,
    PathRAGQueryNode,
    PathRAGRankedPath,
    PathRAGRetrievalContext,
    PathRAGSourceMapping,
    PathRAGTextualizedPath,
    RankedLongUnit,
    RAPTORLayerMembership,
    RAPTORNode,
    RAPTORSourceSegment,
    RAPTORSummarizationProvenance,
    RAPTORTreeCatalog,
    RetrievedHippoRAG2Passage,
    RetrievedLongUnit,
    RetrievedRAPTORNode,
)
from sah_rag.canonical.pilot_adapter import (
    PilotAdapterExampleMapping,
    PilotAdapterManifest,
    PilotAdapterValidation,
)
from evaluation import (
    RetrievalEvaluationCase,
    RetrievalEvaluationRun,
)
from evaluation.contracts import (
    AggregateCutoffMetrics,
    CommonRetrievalEvaluationManifest,
    CommonRetrievalEvaluationRun,
    QueryCutoffMetrics,
)
from orchestration.contracts import (
    ArchitectureExecutionManifest,
    DatasetInputEnvelope,
    ExecutionArtifactRecord,
    ExecutionFailure,
    ExecutionResourceUsage,
    ExperimentCellIdentity,
    ReadinessRegistryEntry,
)
from sah_rag.generation import (
    ContextPack,
    GeneratedAnswer,
    GenerationPrompt,
)
from orchestration import (
    OrchestratorArtifactLayout,
    OrchestratorDryRunPlan,
    OrchestratorExecutionRequest,
    OrchestratorPlanManifest,
    OrchestratorStagePlan,
)
from datasets.pilot_synthetic import (
    PilotCanonicalChunk,
    PilotCanonicalDocument,
    PilotCanonicalJudgment,
    PilotCanonicalQuery,
    PilotDatasetManifest,
)
from sah_rag.retrieval.document_resolution import (
    RetrievedDocument,
)
from sah_rag.storage import (
    QdrantCollectionManifest,
)

SCHEMAS_ROOT = Path("schemas")


def _select_schemas(
    schemas: dict[str, dict],
    only: list[str] | None,
) -> dict[str, dict]:
    if not only:
        return schemas
    selected: dict[str, dict] = {}
    for item in only:
        key = item if item.endswith(".schema.json") else item + ".schema.json"
        if key not in schemas:
            raise SystemExit("Unknown schema: " + item)
        selected[key] = schemas[key]
    return selected


def _serialize(schema: dict) -> bytes:
    return (
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8").replace(b"\r\n", b"\n")


def main(argv: list[str] | None = None) -> int:
    """Write canonical model schemas."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    SCHEMAS_ROOT.mkdir(parents=True, exist_ok=True)

    schemas = {
    "longrag_source_segment.schema.json": (
        LongRAGSourceSegment.model_json_schema(mode="validation")
    ),
    "longrag_long_unit.schema.json": (
        LongUnit.model_json_schema(mode="validation")
    ),
    "longrag_ranked_long_unit.schema.json": (
        RankedLongUnit.model_json_schema(mode="validation")
    ),
    "longrag_construction_summary.schema.json": (
        LongUnitConstructionSummary.model_json_schema(mode="validation")
    ),
    "longrag_embedding_record.schema.json": (
        LongUnitEmbeddingRecord.model_json_schema(mode="validation")
    ),
    "longrag_embedding_shard_header.schema.json": (
        LongUnitEmbeddingShardHeader.model_json_schema(mode="validation")
    ),
    "longrag_embedding_catalog.schema.json": (
        LongUnitEmbeddingCatalog.model_json_schema(mode="validation")
    ),
    "longrag_collection_manifest.schema.json": (
        LongRAGCollectionManifest.model_json_schema(mode="validation")
    ),
    "longrag_retrieved_unit.schema.json": (
        RetrievedLongUnit.model_json_schema(mode="validation")
    ),
    "raptor_source_segment.schema.json": (
        RAPTORSourceSegment.model_json_schema(mode="validation")
    ),
    "raptor_summarization_provenance.schema.json": (
        RAPTORSummarizationProvenance.model_json_schema(mode="validation")
    ),
    "raptor_node.schema.json": (
        RAPTORNode.model_json_schema(mode="validation")
    ),
    "raptor_layer_membership.schema.json": (
        RAPTORLayerMembership.model_json_schema(mode="validation")
    ),
    "raptor_tree_catalog.schema.json": (
        RAPTORTreeCatalog.model_json_schema(mode="validation")
    ),
    "raptor_retrieved_node.schema.json": (
        RetrievedRAPTORNode.model_json_schema(mode="validation")
    ),
    "raptor_expanded_document.schema.json": (
        ExpandedRAPTORDocument.model_json_schema(mode="validation")
    ),
    "hipporag2_canonical_source.schema.json": (
        HippoRAG2CanonicalSource.model_json_schema(mode="validation")
    ),
    "hipporag2_canonical_mapping.schema.json": (
        HippoRAG2CanonicalMapping.model_json_schema(mode="validation")
    ),
    "hipporag2_ner_result.schema.json": (
        HippoRAG2NerResult.model_json_schema(mode="validation")
    ),
    "hipporag2_triple.schema.json": (
        HippoRAG2Triple.model_json_schema(mode="validation")
    ),
    "hipporag2_extraction_result.schema.json": (
        HippoRAG2ExtractionResult.model_json_schema(mode="validation")
    ),
    "hipporag2_openie_catalog.schema.json": (
        HippoRAG2OpenIECatalog.model_json_schema(mode="validation")
    ),
    "hipporag2_graph_node.schema.json": (
        HippoRAG2GraphNode.model_json_schema(mode="validation")
    ),
    "hipporag2_graph_edge.schema.json": (
        HippoRAG2GraphEdge.model_json_schema(mode="validation")
    ),
    "hipporag2_graph_catalog.schema.json": (
        HippoRAG2GraphCatalog.model_json_schema(mode="validation")
    ),
    "hipporag2_fact_score.schema.json": (
        HippoRAG2FactScore.model_json_schema(mode="validation")
    ),
    "hipporag2_recognition_artifact.schema.json": (
        HippoRAG2RecognitionArtifact.model_json_schema(mode="validation")
    ),
    "hipporag2_ppr_seed.schema.json": (
        HippoRAG2PPRSeed.model_json_schema(mode="validation")
    ),
    "hipporag2_ppr_result.schema.json": (
        HippoRAG2PPRResult.model_json_schema(mode="validation")
    ),
    "hipporag2_retrieved_passage.schema.json": (
        RetrievedHippoRAG2Passage.model_json_schema(mode="validation")
    ),
    "hipporag2_expanded_document.schema.json": (
        ExpandedHippoRAG2Document.model_json_schema(mode="validation")
    ),
    "lightrag_canonical_source.schema.json": (
        LightRAGCanonicalSource.model_json_schema(mode="validation")
    ),
    "lightrag_chunk_mapping.schema.json": (
        LightRAGChunkMapping.model_json_schema(mode="validation")
    ),
    "lightrag_extraction_provenance.schema.json": (
        LightRAGExtractionProvenance.model_json_schema(mode="validation")
    ),
    "lightrag_entity.schema.json": (
        LightRAGEntity.model_json_schema(mode="validation")
    ),
    "lightrag_relationship.schema.json": (
        LightRAGRelationship.model_json_schema(mode="validation")
    ),
    "lightrag_extraction_catalog.schema.json": (
        LightRAGExtractionCatalog.model_json_schema(mode="validation")
    ),
    "lightrag_structured_evidence.schema.json": (
        LightRAGStructuredEvidence.model_json_schema(mode="validation")
    ),
    "lightrag_ranked_chunk.schema.json": (
        LightRAGRankedChunk.model_json_schema(mode="validation")
    ),
    "lightrag_expanded_document.schema.json": (
        ExpandedLightRAGDocument.model_json_schema(mode="validation")
    ),
    "lightrag_query_context.schema.json": (
        LightRAGQueryContextArtifact.model_json_schema(mode="validation")
    ),
    "pathrag_canonical_source.schema.json": (
        PathRAGCanonicalSource.model_json_schema(mode="validation")
    ),
    "pathrag_source_mapping.schema.json": (
        PathRAGSourceMapping.model_json_schema(mode="validation")
    ),
    "pathrag_graph_node.schema.json": (
        PathRAGGraphNode.model_json_schema(mode="validation")
    ),
    "pathrag_graph_edge.schema.json": (
        PathRAGGraphEdge.model_json_schema(mode="validation")
    ),
    "pathrag_comparator_catalog.schema.json": (
        PathRAGComparatorCatalog.model_json_schema(mode="validation")
    ),
    "pathrag_query_node.schema.json": (
        PathRAGQueryNode.model_json_schema(mode="validation")
    ),
    "pathrag_candidate_path.schema.json": (
        PathRAGCandidatePath.model_json_schema(mode="validation")
    ),
    "pathrag_propagation_step.schema.json": (
        PathRAGPropagationStep.model_json_schema(mode="validation")
    ),
    "pathrag_propagation_trace.schema.json": (
        PathRAGPropagationTrace.model_json_schema(mode="validation")
    ),
    "pathrag_path_score.schema.json": (
        PathRAGPathScore.model_json_schema(mode="validation")
    ),
    "pathrag_ranked_path.schema.json": (
        PathRAGRankedPath.model_json_schema(mode="validation")
    ),
    "pathrag_textualized_path.schema.json": (
        PathRAGTextualizedPath.model_json_schema(mode="validation")
    ),
    "pathrag_expanded_document.schema.json": (
        ExpandedPathRAGDocument.model_json_schema(mode="validation")
    ),
    "pathrag_retrieval_context.schema.json": (
        PathRAGRetrievalContext.model_json_schema(mode="validation")
    ),
    "experiment_cell_identity.schema.json": (
        ExperimentCellIdentity.model_json_schema(mode="validation")
    ),
    "dataset_input_envelope.schema.json": (
        DatasetInputEnvelope.model_json_schema(mode="validation")
    ),
    "execution_failure.schema.json": (
        ExecutionFailure.model_json_schema(mode="validation")
    ),
    "execution_resource_usage.schema.json": (
        ExecutionResourceUsage.model_json_schema(mode="validation")
    ),
    "execution_artifact_record.schema.json": (
        ExecutionArtifactRecord.model_json_schema(mode="validation")
    ),
    "architecture_execution_manifest.schema.json": (
        ArchitectureExecutionManifest.model_json_schema(mode="validation")
    ),
    "readiness_registry_entry.schema.json": (
        ReadinessRegistryEntry.model_json_schema(mode="validation")
    ),
    "query_cutoff_metrics.schema.json": (
        QueryCutoffMetrics.model_json_schema(mode="validation")
    ),
    "aggregate_cutoff_metrics.schema.json": (
        AggregateCutoffMetrics.model_json_schema(mode="validation")
    ),
    "common_retrieval_evaluation_run.schema.json": (
        CommonRetrievalEvaluationRun.model_json_schema(mode="validation")
    ),
    "common_retrieval_evaluation_manifest.schema.json": (
        CommonRetrievalEvaluationManifest.model_json_schema(mode="validation")
    ),
    "orchestrator_execution_request.schema.json": (
        OrchestratorExecutionRequest.model_json_schema(mode="validation")
    ),
    "orchestrator_stage_plan.schema.json": (
        OrchestratorStagePlan.model_json_schema(mode="validation")
    ),
    "orchestrator_artifact_layout.schema.json": (
        OrchestratorArtifactLayout.model_json_schema(mode="validation")
    ),
    "orchestrator_dry_run_plan.schema.json": (
        OrchestratorDryRunPlan.model_json_schema(mode="validation")
    ),
    "orchestrator_plan_manifest.schema.json": (
        OrchestratorPlanManifest.model_json_schema(mode="validation")
    ),
    "pilot_canonical_document.schema.json": (
        PilotCanonicalDocument.model_json_schema(mode="validation")
    ),
    "pilot_canonical_chunk.schema.json": (
        PilotCanonicalChunk.model_json_schema(mode="validation")
    ),
    "pilot_canonical_query.schema.json": (
        PilotCanonicalQuery.model_json_schema(mode="validation")
    ),
    "pilot_canonical_judgment.schema.json": (
        PilotCanonicalJudgment.model_json_schema(mode="validation")
    ),
    "pilot_dataset_manifest.schema.json": (
        PilotDatasetManifest.model_json_schema(mode="validation")
    ),
    "pilot_adapter_mapping.schema.json": (
        PilotAdapterExampleMapping.model_json_schema(mode="validation")
    ),
    "pilot_adapter_manifest.schema.json": (
        PilotAdapterManifest.model_json_schema(mode="validation")
    ),
    "pilot_adapter_validation.schema.json": (
        PilotAdapterValidation.model_json_schema(mode="validation")
    ),
        "artifact_sidecar_manifest.schema.json": (
            ArtifactSidecarManifest.model_json_schema(mode="validation")
        ),
        "qdrant_collection_manifest.schema.json": (
            QdrantCollectionManifest.model_json_schema(mode="validation")
        ),
        "retrieval_evaluation_case.schema.json": (
            RetrievalEvaluationCase.model_json_schema(mode="validation")
        ),
        "retrieval_evaluation_run.schema.json": (
            RetrievalEvaluationRun.model_json_schema(mode="validation")
        ),
        "generation_context_pack.schema.json": (
            ContextPack.model_json_schema(mode="validation")
        ),
        "generation_prompt.schema.json": (
            GenerationPrompt.model_json_schema(mode="validation")
        ),
        "generated_answer.schema.json": (
            GeneratedAnswer.model_json_schema(mode="validation")
        ),
        "pilot_retrieved_document.schema.json": (
            RetrievedDocument.model_json_schema(mode="validation")
        ),
        "pilot_retrieval_case.schema.json": (
            RetrievalEvaluationCase.model_json_schema(mode="validation")
        ),
        "pilot_query_cutoff_metrics.schema.json": (
            QueryCutoffMetrics.model_json_schema(mode="validation")
        ),
        "pilot_aggregate_cutoff_metrics.schema.json": (
            AggregateCutoffMetrics.model_json_schema(mode="validation")
        ),
        "pilot_evaluation_execution_manifest.schema.json": (
            CommonRetrievalEvaluationManifest.model_json_schema(mode="validation")
        ),
    }

    selected = _select_schemas(schemas, args.only)

    exit_code = 0
    for name, schema in selected.items():
        destination = SCHEMAS_ROOT / name
        expected = _serialize(schema)

        if args.check_only:
            if not destination.exists():
                print(f"[FAIL] Missing: {name}")
                exit_code = 1
                continue
            existing = destination.read_bytes().replace(b"\r\n", b"\n")
            if existing != expected:
                print(f"[DIFF] {name}")
                exit_code = 1
            else:
                print(f"[OK] {name}")
            continue

        destination.write_bytes(expected)
        print(f"[OK] Exported {destination.name}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
