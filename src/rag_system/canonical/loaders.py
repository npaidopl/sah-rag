"""Typed streaming loaders for canonical datasets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from rag_system.canonical.jsonl import iter_jsonl
from rag_system.canonical.models import (
    CanonicalDocument,
    CanonicalExample,
)
from rag_system.canonical.paths import (
    CanonicalDatasetPaths,
    resolve_dataset_paths,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class CanonicalModelValidationError(ValueError):
    """Raised when a canonical record fails model validation."""

    def __init__(
        self,
        *,
        path: Path,
        record_number: int,
        model_name: str,
        validation_error: ValidationError,
    ) -> None:
        self.path = path
        self.record_number = record_number
        self.model_name = model_name
        self.validation_error = validation_error

        super().__init__(
            f"{path}: record {record_number} failed "
            f"{model_name} validation: {validation_error}"
        )


def iter_validated_models[ModelT: BaseModel](
    path: Path,
    model_type: type[ModelT],
) -> Iterator[ModelT]:
    """Stream JSONL records and validate them as typed models."""

    for record_number, record in enumerate(
        iter_jsonl(path),
        start=1,
    ):
        try:
            yield model_type.model_validate(record)
        except ValidationError as exc:
            raise CanonicalModelValidationError(
                path=path,
                record_number=record_number,
                model_name=model_type.__name__,
                validation_error=exc,
            ) from exc


def iter_canonical_documents(
    dataset: str | CanonicalDatasetPaths,
) -> Iterator[CanonicalDocument]:
    """Stream validated documents for one canonical dataset."""

    paths = (
        resolve_dataset_paths(dataset)
        if isinstance(dataset, str)
        else dataset
    )

    yield from iter_validated_models(
        paths.documents,
        CanonicalDocument,
    )


def iter_canonical_examples(
    dataset: str | CanonicalDatasetPaths,
) -> Iterator[CanonicalExample]:
    """Stream validated benchmark examples for one dataset."""

    paths = (
        resolve_dataset_paths(dataset)
        if isinstance(dataset, str)
        else dataset
    )

    yield from iter_validated_models(
        paths.examples,
        CanonicalExample,
    )