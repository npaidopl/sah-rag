"""Guard-rail tests for the Step 92 registration script."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "register_external_canonical_suite.py"


def test_registration_script_does_not_load_model_or_tokenizer():
    src = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "BGEM3FlagModel",
        "FlagEmbedding",
        "AutoTokenizer",
        "from_pretrained",
        "SentencePiece",
    ):
        assert token not in src, token


def test_registration_script_does_not_execute_pipeline():
    src = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "run_pilot_parser_chunker(",
        "run_pilot_embedding_materialization(",
        "run_pilot_qdrant_ingestion(",
        "run_pilot_hybrid_retrieval(",
        "run_pilot_retrieval_evaluation(",
        "QdrantClient",
    ):
        assert token not in src, token


def test_registration_script_does_not_write_outside_audit():
    src = SCRIPT.read_text(encoding="utf-8")
    # every write must target one of the three audit YAMLs
    for line in src.splitlines():
        if "write_text(" in line:
            assert any(
                target in line
                for target in ("REGISTRY_PATH", "MANIFEST_PATH", "CONTRACT_PATH")
            ), line


def test_registration_streams_jsonl():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "for line in fh" in src
    # no full list of records retained
    assert "list(handle)" not in src