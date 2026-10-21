"""Atomic reading and writing of artifact manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def write_json_model_atomic(
    path: Path,
    model: BaseModel,
    *,
    overwrite: bool = False,
) -> None:
    """Write one Pydantic model as atomic JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}"
        )

    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                model.model_dump(mode="json"),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temporary_path,
            path,
        )
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


def read_json_model[ModelT: BaseModel](
    path: Path,
    model_type: type[ModelT],
) -> ModelT:
    """Read and validate one JSON model."""

    if not path.is_file():
        raise FileNotFoundError(
            f"JSON model file does not exist: {path}"
        )

    raw_text = path.read_text(encoding="utf-8-sig")
    return model_type.model_validate_json(raw_text)
