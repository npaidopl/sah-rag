"""Direct canonical loaders (bypass RAG_CANONICAL_ROOT env var)."""
import json
from pathlib import Path
from typing import Iterator

from sah_rag.canonical.models import CanonicalDocument
from ablation.config import CANONICAL_DATA_ROOT


def iter_canonical_documents_local(dataset: str) -> Iterator[CanonicalDocument]:
    path = CANONICAL_DATA_ROOT / dataset / "documents.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield CanonicalDocument.model_validate(json.loads(line))

from sah_rag.canonical.models import CanonicalExample
from typing import Iterator


def iter_canonical_examples_local(dataset: str) -> Iterator[CanonicalExample]:
    path = CANONICAL_DATA_ROOT / dataset / "examples.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield CanonicalExample.model_validate(json.loads(line))
