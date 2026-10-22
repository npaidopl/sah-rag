"""Persistence smoke for the pilot parser+chunker runner."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_pilot_parser_chunker.py"
RAG_ROOT = (
    REPO_ROOT
    / "outputs" / "infrastructure_pilot"
    / "infrastructure-pilot-synthetic-v1"
    / "sah_rag"
)
# The parser+chunker runner writes only these two subdirectories.
# We must never delete the entire sah_rag/ tree: sibling
# directories (embedding/, qdrant/) belong to later pipeline stages.
PARSER_CHUNKER_SUBDIRS = ("parsing", "chunking")


def _clean_parser_chunker_outputs() -> None:
    """Delete only the artifacts produced by this runner."""
    for name in PARSER_CHUNKER_SUBDIRS:
        target = RAG_ROOT / name
        if target.exists() and target.is_dir():
            shutil.rmtree(target)


def _tree_sha(root: Path) -> str:
    """Hash only the deterministic data artifacts.

    Execution manifests contain runtime metadata (timestamps, git
    revision) and are intentionally excluded from the determinism
    comparison.
    """
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.jsonl")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


def _parser_chunker_sha() -> str:
    """Hash only the parser+chunker subtree, not the whole sah_rag/."""
    h = hashlib.sha256()
    for name in PARSER_CHUNKER_SUBDIRS:
        subtree = RAG_ROOT / name
        if not subtree.exists():
            continue
        for p in sorted(subtree.rglob("*.jsonl")):
            if p.is_file():
                h.update(str(p.relative_to(RAG_ROOT)).encode("utf-8"))
                h.update(p.read_bytes())
    return h.hexdigest()


def _run(overwrite: bool) -> int:
    cmd = [sys.executable, str(RUNNER)]
    if overwrite:
        cmd.append("--overwrite")
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return r.returncode


def test_round_trip_succeeds():
    _clean_parser_chunker_outputs()
    assert _run(overwrite=False) == 0


def test_existing_outputs_rejected_without_overwrite():
    assert _run(overwrite=False) != 0


def test_overwrite_succeeds():
    assert _run(overwrite=True) == 0


def test_deterministic_across_two_runs():
    _run(overwrite=True)
    a = _parser_chunker_sha()
    _run(overwrite=True)
    b = _parser_chunker_sha()
    assert a == b


def test_source_artifacts_unchanged():
    src1 = REPO_ROOT / "data" / "pilot" / "infrastructure-pilot-synthetic-v1"
    src2 = REPO_ROOT / "processed" / "canonical" / "infrastructure-pilot-synthetic-v1"
    before = (_tree_sha(src1), _tree_sha(src2))
    _run(overwrite=True)
    after = (_tree_sha(src1), _tree_sha(src2))
    assert before == after


def test_embedding_subtree_untouched():
    """Guarantee: this suite never touches sibling embedding/ subtree.

    Step 83 embeddings are an upstream dependency of Step 85 Qdrant
    ingestion. The parser+chunker persistence suite must not delete
    them, even indirectly.
    """
    embedding_root = RAG_ROOT / "embedding"
    existed_before = embedding_root.exists()

    _clean_parser_chunker_outputs()
    _run(overwrite=True)

    existed_after = embedding_root.exists()
    assert existed_after == existed_before, (
        "parser+chunker suite must not touch the embedding subtree"
    )