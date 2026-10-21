"""Derived artifact models and utilities."""

from rag_system.artifacts.fingerprints import (
    count_nonempty_lines,
    file_size_bytes,
    sha256_file,
)
from rag_system.artifacts.identity import (
    build_chunk_id,
    build_experiment_id,
    build_parent_id,
    canonical_json_bytes,
    combine_fingerprints,
    normalize_identifier,
    sha256_fingerprint,
    short_fingerprint,
)
from rag_system.artifacts.jsonl import (
    AtomicJsonlWriter,
    serialize_record,
    write_jsonl_atomic,
)
from rag_system.artifacts.manifest import (
    ArtifactReference,
    ComponentReference,
    ExecutionEnvironment,
    ExecutionReference,
    ExperimentManifest,
    ExperimentStatus,
)
from rag_system.artifacts.manifest_io import (
    read_json_model,
    write_json_model_atomic,
)
from rag_system.artifacts.models import (
    ArtifactBaseModel,
    BoundaryEvidence,
    BoundaryMethod,
    ChildChunk,
    OffsetSpan,
    ParsedDocument,
    SemanticParent,
)
from rag_system.artifacts.revision import (
    get_git_revision,
)
from rag_system.artifacts.sidecar import (
    ArtifactComponentReference,
    ArtifactInputReference,
    ArtifactSidecarManifest,
    build_file_sidecar_manifest,
    build_input_reference,
    build_jsonl_sidecar_manifest,
)

__all__ = [
    "ArtifactBaseModel",
    "ArtifactComponentReference",
    "ArtifactInputReference",
    "ArtifactReference",
    "ArtifactSidecarManifest",
    "AtomicJsonlWriter",
    "BoundaryEvidence",
    "BoundaryMethod",
    "ChildChunk",
    "ComponentReference",
    "ExecutionEnvironment",
    "ExecutionReference",
    "ExperimentManifest",
    "ExperimentStatus",
    "OffsetSpan",
    "ParsedDocument",
    "SemanticParent",
    "build_chunk_id",
    "build_experiment_id",
    "build_input_reference",
    "build_file_sidecar_manifest",
    "build_jsonl_sidecar_manifest",
    "build_parent_id",
    "canonical_json_bytes",
    "combine_fingerprints",
    "count_nonempty_lines",
    "file_size_bytes",
    "get_git_revision",
    "normalize_identifier",
    "read_json_model",
    "serialize_record",
    "sha256_file",
    "sha256_fingerprint",
    "short_fingerprint",
    "write_json_model_atomic",
    "write_jsonl_atomic",
]
