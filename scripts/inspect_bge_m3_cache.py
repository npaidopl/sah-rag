"""Inspect the local BGE-M3 HF cache against the frozen revision."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

MODEL_DIR_NAME = "models--BAAI--bge-m3"
FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "pytorch_model.bin",
    "sparse_linear.pt",
    "colbert_linear.pt",
    "sentence_bert_config.json",
    "modules.json",
)


def _resolve_cache_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    load_dotenv(dotenv_path=Path(".env"), override=False)
    for key in ("HF_HOME", "RAG_HF_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.getenv(key)
        if value:
            return Path(value)
    raise RuntimeError("no HF cache root configured")


def _snapshot_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def inspect(cache_root: Path) -> dict:
    model_dir = cache_root / MODEL_DIR_NAME
    report: dict = {
        "cache_root": str(cache_root),
        "model_dir": str(model_dir),
        "model_dir_present": model_dir.is_dir(),
        "frozen_revision": FROZEN_REVISION,
    }

    if not model_dir.is_dir():
        report["cache_status"] = "NOT_AVAILABLE"
        report["embedding_execution_allowed"] = False
        return report

    refs_main = model_dir / "refs" / "main"
    if refs_main.is_file():
        report["refs_main"] = refs_main.read_text(encoding="utf-8").strip()
    else:
        report["refs_main"] = None

    snapshot = model_dir / "snapshots" / FROZEN_REVISION
    report["snapshot_path"] = str(snapshot)
    report["snapshot_present"] = snapshot.is_dir()

    if not snapshot.is_dir():
        report["cache_status"] = "REVISION_MISSING"
        report["embedding_execution_allowed"] = False
        return report

    missing: list[str] = []
    for name in REQUIRED_FILES:
        if not (snapshot / name).is_file():
            missing.append(name)
    report["missing_files"] = missing
    report["required_files"] = list(REQUIRED_FILES)

    report["snapshot_size_bytes"] = _snapshot_size(snapshot)

    locks = list(model_dir.rglob("*.lock"))
    report["lock_files_present"] = bool(locks)
    report["lock_files"] = [str(p) for p in locks]

    ok = not missing and not locks
    report["cache_status"] = "VERIFIED" if ok else "PARTIAL"
    report["embedding_execution_allowed"] = ok

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cache_root = _resolve_cache_root(args.cache_root)
    report = inspect(cache_root)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    print("[OK] BGE-M3 cache inspection")
    print(f"  cache root: {report['cache_root']}")
    print(f"  frozen revision: {report['frozen_revision']}")
    print(f"  refs/main: {report.get('refs_main')}")
    print(f"  snapshot present: {report.get('snapshot_present')}")
    print(f"  missing files: {report.get('missing_files', [])}")
    print(f"  locks: {report.get('lock_files_present')}")
    print(f"  snapshot size bytes: {report.get('snapshot_size_bytes')}")
    print(f"  cache_status: {report['cache_status']}")
    print(f"  embedding_execution_allowed: {report['embedding_execution_allowed']}")

    return 0 if report["cache_status"] == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
