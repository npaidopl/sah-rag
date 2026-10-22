"""Base contract for all SAH-RAG baselines (v2).

Every baseline in `baselines/<name>/adapter.py` MUST subclass `BaseBaseline`.
The orchestrator instantiates baselines via `BaseBaseline.__init__()` and
calls `build_index()` then `retrieve()`.

This module is FROZEN. Do not modify without a version bump.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class BaselineStatus(str, Enum):
    """Where the baseline lives in the execution lifecycle."""
    EXECUTABLE = "executable"    # we have code and can run it
    REPORTED = "reported"        # paper-reported numbers only
    SKIPPED = "skipped"          # explicitly excluded (license, deprecated)


class RunMode(str, Enum):
    """Which mode the orchestrator invokes the baseline in."""
    NATIVE = "native"
    CONTROLLED = "controlled"


@dataclass(frozen=True)
class RetrievedChunk:
    """Standardized retrieval output. Every baseline MUST return these."""
    doc_id: str
    chunk_id: str
    text: str
    score: float
    rank: int                        # 1-indexed
    parent_doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Embedder(Protocol):
    """Injected by orchestrator in controlled mode."""
    @property
    def name(self) -> str: ...
    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...
    def encode_query(self, text: str) -> list[float]: ...


class LLM(Protocol):
    """Injected by orchestrator in controlled mode."""
    @property
    def name(self) -> str: ...
    def generate(self, prompt: str, **kwargs: Any) -> str: ...


@dataclass
class BaselineConfig:
    """Parsed from config_native.yaml or config_controlled.yaml."""
    name: str
    mode: RunMode
    top_k: int = 10
    extra: dict[str, Any] = field(default_factory=dict)


class BaseBaseline(ABC):
    """Every baseline in `baselines/<name>/adapter.py` subclasses this."""

    # --- Identity (MUST be overridden) ---
    name: str
    version: str
    status: BaselineStatus = BaselineStatus.EXECUTABLE
    supports_native: bool = False
    supports_controlled: bool = False
    is_llm_based: bool = False

    def __init__(
        self,
        *,
        config: BaselineConfig,
        work_dir: Path,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.llm = llm
        self._validate()

    def _validate(self) -> None:
        """Fail-fast if mode requirements aren't met."""
        if self.status != BaselineStatus.EXECUTABLE:
            return
        mode = self.config.mode
        if mode == RunMode.CONTROLLED:
            if not self.supports_controlled:
                raise ValueError(f"{self.name}: does not support controlled mode")
            if self.embedder is None:
                raise ValueError(f"{self.name}: controlled mode requires embedder injection")
            if self.is_llm_based and self.llm is None:
                raise ValueError(
                    f"{self.name}: LLM-based baseline requires llm injection in controlled mode"
                )
        elif mode == RunMode.NATIVE:
            if not self.supports_native:
                raise ValueError(f"{self.name}: does not support native mode")

    # --- Lifecycle (MUST be overridden) ---
    @abstractmethod
    def build_index(self, documents: list[Any]) -> None:
        """Build the retrieval index from the corpus.

        `documents` is a list of `sah_rag.canonical.models.CanonicalDocument`.
        Persist everything needed for `retrieve()` under `self.work_dir`.
        Must be idempotent: calling twice on same corpus is safe.
        """
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a single query.

        Returns list of RetrievedChunk, ranks 1..k contiguous.
        If `top_k` is None, use `self.config.top_k`.
        """
        ...

    # --- Optional hooks ---
    def cleanup(self) -> None:
        """Called after evaluation. Free resources."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} status={self.status.value}>"


class ReportedBaseline(BaseBaseline):
    """Baseline with no code — returns paper-reported metrics only."""

    def build_index(self, documents: list[Any]) -> None:
        pass

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        raise NotImplementedError(
            f"{self.name} is reported-only. Use `get_reported_metrics()`."
        )

    @abstractmethod
    def get_reported_metrics(self, dataset_id: str) -> dict[str, float]:
        """Return metric_name -> value, as reported in the paper."""
        ...
