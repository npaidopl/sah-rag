"""Canonical dataset integration package."""

from rag_system.canonical.loaders import (
    CanonicalModelValidationError,
    iter_canonical_documents,
    iter_canonical_examples,
    iter_validated_models,
)
from rag_system.canonical.models import (
    CanonicalDocument,
    CanonicalExample,
    CanonicalGroundTruth,
    CanonicalQuery,
    GroundTruthType,
)
from rag_system.canonical.paths import (
    CanonicalDatasetPaths,
    get_canonical_root,
    list_dataset_ids,
    resolve_dataset_paths,
)

__all__ = [
    "CanonicalDatasetPaths",
    "CanonicalDocument",
    "CanonicalExample",
    "CanonicalGroundTruth",
    "CanonicalModelValidationError",
    "CanonicalQuery",
    "GroundTruthType",
    "get_canonical_root",
    "iter_canonical_documents",
    "iter_canonical_examples",
    "iter_validated_models",
    "list_dataset_ids",
    "resolve_dataset_paths",
]