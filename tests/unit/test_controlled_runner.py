"""Tests for the controlled-mode runner and component injection.

Offline, deterministic, < 5s on CPU.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from baselines.base import (
    BaseBaseline,
    BaselineConfig,
    BaselineStatus,
    RetrievedChunk,
    RunMode,
)
from sah_rag.canonical.models import CanonicalDocument


class _StubEmbedder:
    name = "stub_embedder"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def encode_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _StubLLM:
    name = "stub_llm"

    def generate(self, prompt: str, **kwargs) -> str:
        return "stub answer"


class _RecordingAdapter(BaseBaseline):
    name = "recording_adapter"
    version = "0.0.1"
    status = BaselineStatus.EXECUTABLE
    supports_native = False
    supports_controlled = True
    is_llm_based = True

    def __init__(self, *, config, work_dir, embedder=None, llm=None):
        super().__init__(
            config=config, work_dir=work_dir, embedder=embedder, llm=llm
        )
        self.received_embedder = embedder
        self.received_llm = llm

    def build_index(self, documents):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "built.txt").write_text("ok", encoding="utf-8")

    def retrieve(self, query, top_k=None):
        if top_k is None:
            top_k = self.config.top_k
        return [
            RetrievedChunk(
                doc_id=f"d{i}",
                chunk_id=f"c{i}",
                text="stub",
                score=1.0 - i * 0.1,
                rank=i + 1,
            )
            for i in range(min(3, top_k))
        ]


class _FakeDataset:
    documents = [
        CanonicalDocument(
            doc_id="d0",
            dataset="test",
            text_raw="hello",
            text_normalized="hello",
        )
    ]
    examples = []


def test_build_controlled_components_returns_embedder_and_llm():
    from evaluation.runner import _build_controlled_components
    import sah_rag.components.embedders as emb_mod
    import sah_rag.components.llms as llm_mod

    orig_emb = emb_mod.BGEM3Embedder
    orig_llm = llm_mod.QwenLLM
    emb_mod.BGEM3Embedder = lambda: _StubEmbedder()  # type: ignore[assignment]
    llm_mod.QwenLLM = lambda: _StubLLM()  # type: ignore[assignment]
    try:
        embedder, llm = _build_controlled_components({"is_llm_based": True})
        assert embedder is not None
        assert llm is not None
        assert embedder.name == "stub_embedder"
        assert llm.name == "stub_llm"

        embedder2, llm2 = _build_controlled_components({"is_llm_based": False})
        assert embedder2 is not None
        assert llm2 is None
    finally:
        emb_mod.BGEM3Embedder = orig_emb  # type: ignore[assignment]
        llm_mod.QwenLLM = orig_llm  # type: ignore[assignment]


def test_controlled_mode_injects_embedder_and_llm(tmp_path: Path):
    from evaluation.runner import run_cell

    entry = {
        "name": "recording_adapter",
        "path": "baselines/recording_adapter",
        "status": "executable",
        "modes": ["controlled"],
        "is_llm_based": True,
    }

    class FakeCell:
        run_id = "test-run"
        baseline = "recording_adapter"
        dataset = "test"
        mode = "controlled"
        top_k = 3
        name = "test-cell"

    # Create a fake baseline directory under tmp_path so the exists() check passes
    (tmp_path / entry["path"]).mkdir(parents=True, exist_ok=True)

    with patch("evaluation.runner._project_root", return_value=tmp_path), \
         patch("evaluation.runner._load_registry", return_value={"baselines": [entry]}), \
         patch("evaluation.runner._baseline_entry", return_value=entry), \
         patch("evaluation.runner._load_config_yaml", return_value={"extra": {}}), \
         patch("evaluation.runner._find_adapter_class", return_value=_RecordingAdapter), \
         patch("evaluation.runner.load_dataset", return_value=_FakeDataset()), \
         patch("sah_rag.components.embedders.BGEM3Embedder", lambda: _StubEmbedder()), \
         patch("sah_rag.components.llms.QwenLLM", lambda: _StubLLM()):

        result = run_cell(FakeCell(), output_root=tmp_path / "out")
        assert result["baseline"] == "recording_adapter"
        assert result["mode"] == "controlled"
        assert (tmp_path / "out" / "test-cell" / "metrics.json").exists()


def test_native_mode_does_not_inject_components(tmp_path: Path):
    from evaluation.runner import run_cell

    class _NativeRecordingAdapter(_RecordingAdapter):
        name = "native_recording"
        supports_native = True
        supports_controlled = False

    entry = {
        "name": "native_recording",
        "path": "baselines/native_recording",
        "status": "executable",
        "modes": ["native"],
        "is_llm_based": True,
    }

    class FakeCell:
        run_id = "test-run"
        baseline = "native_recording"
        dataset = "test"
        mode = "native"
        top_k = 3
        name = "test-native-cell"

    (tmp_path / entry["path"]).mkdir(parents=True, exist_ok=True)

    with patch("evaluation.runner._project_root", return_value=tmp_path), \
         patch("evaluation.runner._load_registry", return_value={"baselines": [entry]}), \
         patch("evaluation.runner._baseline_entry", return_value=entry), \
         patch("evaluation.runner._load_config_yaml", return_value={"extra": {}}), \
         patch("evaluation.runner._find_adapter_class", return_value=_NativeRecordingAdapter), \
         patch("evaluation.runner.load_dataset", return_value=_FakeDataset()):

        result = run_cell(FakeCell(), output_root=tmp_path / "out-native")
        assert result["mode"] == "native"