"""Sequential, resumable scheduler for SAH-RAG v2 evaluation runs."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_env_loaded() -> None:
    """Load .env from project root once if SAH_RAG_EXPERIMENTS_DIR is missing."""
    if os.getenv("SAH_RAG_EXPERIMENTS_DIR"):
        return
    env_file = _PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    except ImportError:
        pass


_ensure_env_loaded()

from orchestration.index import RunIndex  # noqa: E402
from orchestration.manifest import RunManifest, load_manifest  # noqa: E402


def _output_root(manifest: RunManifest) -> Path:
    if manifest.output_root:
        return Path(manifest.output_root)
    env = os.getenv("SAH_RAG_EXPERIMENTS_DIR")
    if not env:
        raise RuntimeError(
            "Neither output_root nor SAH_RAG_EXPERIMENTS_DIR set. "
            f"Create {_PROJECT_ROOT / '.env'} from .env.example."
        )
    return Path(env) / "runs"


def _run_cell(cell, output_root: Path) -> dict:
    from evaluation.runner import run_cell as eval_run_cell
    return eval_run_cell(cell, output_root)


def run_manifest(manifest_path: Path, *, dry_run: bool = False, force: bool = False) -> dict:
    manifest = load_manifest(manifest_path)
    output_root = _output_root(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    index = RunIndex(output_root / "index.db")
    cells = manifest.cells()
    print(f"Manifest: {manifest_path}")
    print(f"Cells: {len(cells)} -> {output_root}")
    print(f"Modes: {manifest.modes} | Baselines: {manifest.baselines} | Datasets: {manifest.datasets}")
    print()
    counts = {"done": 0, "skipped": 0, "failed": 0, "dry": 0}
    try:
        for i, cell in enumerate(cells, 1):
            prefix = f"[{i}/{len(cells)}] {cell.name}"
            if not force and index.is_done(cell):
                print(f"{prefix} -> SKIP (done)"); counts["skipped"] += 1; continue
            if dry_run:
                print(f"{prefix} -> DRY-RUN"); counts["dry"] += 1; continue
            run_dir = output_root / cell.name
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"{prefix} -> START")
            index.mark_started(cell)
            try:
                metrics = _run_cell(cell, output_root)
                index.mark_done(cell, json.dumps(metrics))
                (run_dir / "DONE").write_text("ok", encoding="utf-8")
                print(f"{prefix} -> DONE ({metrics.get('n_evaluated', 0)} examples)")
                counts["done"] += 1
            except Exception as exc:
                tb = traceback.format_exc()
                (run_dir / "error.log").write_text(tb, encoding="utf-8")
                index.mark_failed(cell, str(exc))
                print(f"{prefix} -> FAILED: {exc}")
                counts["failed"] += 1
    finally:
        index.close()
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orchestration.scheduler")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    counts = run_manifest(args.manifest, dry_run=args.dry_run, force=args.force)
    print()
    print(f"Summary: {counts}")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())