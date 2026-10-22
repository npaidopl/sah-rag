"""Derived artifact models and utilities."""

from sah_rag.artifacts.fingerprints import (
    count_nonempty_lines,
    file_size_bytes,
    sha256_file,
)
from sah_rag.artifacts.identity import (
    build_chunk_id,
    build_experiment_id,
    build_parent_id,
    canonical_json_bytes,
    combine_fingerprints,
    normalize_identifier,
    sha256_fingerprint,
    short_fingerprint,
)
from sah_rag.artifacts.jsonl import (
    AtomicJsonlWriter,
    serialize_record,
    write_jsonl_atomic,
)
from sah_rag.artifacts.manifest import (
    ArtifactReference,
    ComponentReference,
    ExecutionEnvironment,
    ExecutionReference,
    ExperimentManifest,
    ExperimentStatus,
)
from sah_rag.artifacts.manifest_io import (
    read_json_model,
    write_json_model_atomic,
)
from sah_rag.artifacts.models import (
    ArtifactBaseModel,
    BoundaryEvidence,
    BoundaryMethod,
    ChildChunk,
    OffsetSpan,
    ParsedDocument,
    SemanticParent,
)
from sah_rag.artifacts.revision import (
    get_git_revision,
)
from sah_rag.artifacts.sidecar import (
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
