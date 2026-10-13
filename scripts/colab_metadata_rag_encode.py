# ruff: noqa: E402
"""Colab-side BGE-M3 dense encoder for Metadata-RAG bundles.

This script runs inside a Google Colab session (or any GPU host) and
consumes the per-dataset bundle produced by
``scripts/run_metadata_rag_state_materialization.py``.

It performs exactly three things:

  1. Extract and checksum-verify the input bundle.
  2. Encode every window text and every query text with BGE-M3
     (dense-only, FP16 on CUDA), writing one record per input:
         {"doc_id": "...", "dense_vector": [...]}      (windows)
         {"query_id": "...", "dense_vector": [...]}    (queries)
  3. Package the encoded arrays into a self-contained tar.gz.

The script intentionally does NOT attach metadata (schema version,
extractor version, metadata record IDs). The local import stage
(``scripts/import_metadata_rag_colab_embeddings.py``) reattaches
those from the frozen state, so the Colab side never needs to know
the metadata schema.

Dry-run mode:
    With ``--dry-run`` the script uses a deterministic contract
    encoder derived from SHA-256 of the input text, with no model
    load. This is only for testing the pipeline contract.

Usage (Colab):
    !pip install -q FlagEmbedding
    !python colab_metadata_rag_encode.py \
        --bundle clerk_bundle.tar.gz \
        --output clerk_embeddings.tar.gz

Usage (local, contract smoke test):
    python colab_metadata_rag_encode.py \
        --bundle <bundle> --output <out> --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import sys
import tarfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Protocol

MODEL_ID_DEFAULT: Final[str] = "BAAI/bge-m3"
MODEL_REVISION_DEFAULT: Final[str] = (
    "5617a9f61b028005a4858fdac845db406aefb181"
)
DENSE_DIMENSION: Final[int] = 1024

BUNDLE_MANIFEST: Final[str] = "manifest.json"
BUNDLE_WINDOWS: Final[str] = "windows.jsonl"
BUNDLE_QUERIES: Final[str] = "queries.jsonl"
BUNDLE_CHECKSUMS: Final[str] = "checksums.json"

OUTPUT_WINDOW_EMBEDDINGS: Final[str] = "window_embeddings.jsonl"
OUTPUT_QUERY_EMBEDDINGS: Final[str] = "query_embeddings.jsonl"
OUTPUT_MANIFEST: Final[str] = "manifest.json"
OUTPUT_CHECKSUMS: Final[str] = "checksums.json"

ENCODING_ARTIFACT_TYPE: Final[str] = "metadata_rag_embeddings"
ENCODING_ARTIFACT_VERSION: Final[str] = "1.0"


class ColabEncodingError(ValueError):
    """Raised when encoding violates its contract."""


# -------------------------------------------------------------------------
# Deterministic I/O
# -------------------------------------------------------------------------


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)


def _write_json(path: Path, obj: Any) -> None:
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    _atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> int:
    parts: list[str] = []
    for row in rows:
        parts.append(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
    payload = ("\n".join(parts) + "\n").encode("utf-8")
    _atomic_write_bytes(path, payload)
    return len(rows)


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ColabEncodingError(f"{path.name} is not a JSON object")
    return parsed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ColabEncodingError(
                f"{path.name}:{lineno} not a JSON object"
            )
        rows.append(parsed)
    return rows


# -------------------------------------------------------------------------
# Bundle extraction
# -------------------------------------------------------------------------


def _extract_bundle(bundle_tar: Path, target_dir: Path) -> dict[str, Any]:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle_tar, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name not in (
                BUNDLE_MANIFEST,
                BUNDLE_WINDOWS,
                BUNDLE_QUERIES,
                BUNDLE_CHECKSUMS,
            ):
                continue
            if member.isfile():
                tar.extract(member, path=target_dir, filter="data")

    checksums = _read_json(target_dir / BUNDLE_CHECKSUMS)
    for name in (BUNDLE_MANIFEST, BUNDLE_WINDOWS, BUNDLE_QUERIES):
        if name not in checksums:
            raise ColabEncodingError(
                f"bundle checksums missing entry for {name}"
            )
        observed = _sha256_file(target_dir / name)
        if observed != checksums[name]:
            raise ColabEncodingError(
                f"bundle checksum mismatch for {name}: "
                f"observed {observed}, declared {checksums[name]}"
            )

    manifest = _read_json(target_dir / BUNDLE_MANIFEST)
    if manifest.get("artifact_type") != "metadata_rag_colab_bundle":
        raise ColabEncodingError(
            "bundle.artifact_type is not metadata_rag_colab_bundle"
        )
    return manifest


# -------------------------------------------------------------------------
# Encoders
# -------------------------------------------------------------------------


class _EncoderProtocol(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[list[float]]: ...


class _ContractEncoder:
    """Deterministic contract encoder for --dry-run.

    Produces a unit-norm 1024-d vector from the SHA-256 of the input
    text. NOT a language model. Only used to exercise the pipeline
    contract without loading BGE-M3.
    """

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[list[float]]:
        del batch_size, max_length
        rows: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = (
                digest
                * (DENSE_DIMENSION // len(digest) + 1)
            )[:DENSE_DIMENSION]
            vec = [float(b) - 128.0 for b in raw]
            norm = math.sqrt(sum(v * v for v in vec))
            if norm == 0.0:
                raise ColabEncodingError(
                    "contract encoder produced a zero-norm vector"
                )
            rows.append([v / norm for v in vec])
        return rows


class _BgeM3DenseEncoder:
    """Thin wrapper around FlagEmbedding BGEM3FlagModel, dense-only."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        device: str,
        use_fp16: bool,
        snapshot_dir: Path | None,
    ) -> None:
        module: Any = importlib.import_module("FlagEmbedding")
        BGEM3FlagModel = module.BGEM3FlagModel

        if snapshot_dir is not None:
            model = BGEM3FlagModel(
                str(snapshot_dir),
                use_fp16=use_fp16,
                device=device,
                local_files_only=True,
            )
        else:
            model = BGEM3FlagModel(
                model_id,
                revision=model_revision,
                use_fp16=use_fp16,
                device=device,
            )
        self._model = model

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[list[float]]:
        out = self._model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        rows: list[list[float]] = []
        for row in dense:
            rows.append([float(x) for x in row])
        return rows


# -------------------------------------------------------------------------
# Encoding
# -------------------------------------------------------------------------


def _encode_in_batches(
    encoder: _EncoderProtocol,
    texts: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
    label: str,
) -> list[list[float]]:
    if batch_size < 1:
        raise ColabEncodingError("batch_size must be >= 1")
    if max_length < 8:
        raise ColabEncodingError("max_length must be >= 8")
    if not texts:
        return []

    rows: list[list[float]] = []
    n = len(texts)
    t0 = time.monotonic()
    for start in range(0, n, batch_size):
        chunk = list(texts[start : start + batch_size])
        out = encoder.encode(
            chunk, batch_size=batch_size, max_length=max_length
        )
        if len(out) != len(chunk):
            raise ColabEncodingError(
                f"{label}: encoder returned {len(out)} rows for "
                f"{len(chunk)} inputs"
            )
        for vec in out:
            if len(vec) != DENSE_DIMENSION:
                raise ColabEncodingError(
                    f"{label}: vector length {len(vec)} != "
                    f"{DENSE_DIMENSION}"
                )
            rows.append(vec)
        elapsed = time.monotonic() - t0
        done = min(start + batch_size, n)
        if done % max(1, (n // 10)) == 0 or done == n:
            print(
                f"  [{label}] {done}/{n} "
                f"({elapsed:.1f}s, {done / max(elapsed, 1e-9):.1f}/s)"
            )
    return rows


# -------------------------------------------------------------------------
# Packaging
# -------------------------------------------------------------------------


def _pack_output(output_dir: Path, target_tar: Path) -> None:
    tmp = target_tar.with_name(target_tar.name + ".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for name in (
            OUTPUT_MANIFEST,
            OUTPUT_WINDOW_EMBEDDINGS,
            OUTPUT_QUERY_EMBEDDINGS,
            OUTPUT_CHECKSUMS,
        ):
            tar.add(output_dir / name, arcname=name)
    tmp.replace(target_tar)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def _build_encoder(args: argparse.Namespace) -> tuple[_EncoderProtocol, str]:
    if args.dry_run:
        return _ContractEncoder(), "DETERMINISTIC_CONTRACT_ENCODER"
    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                print(
                    "[warn] --device cuda requested but CUDA is not "
                    "available; falling back to cpu"
                )
                args.device = "cpu"
        except ImportError:
            print("[warn] torch not importable; falling back to cpu")
            args.device = "cpu"
    enc = _BgeM3DenseEncoder(
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        use_fp16=args.use_fp16 and args.device == "cuda",
        snapshot_dir=args.snapshot_dir,
    )
    return enc, "BGE_M3_DENSE_ONLY"


def run_encode(
    *,
    bundle_tar: Path,
    output_tar: Path,
    work_dir: Path,
    dry_run: bool,
    model_id: str,
    model_revision: str,
    device: str,
    use_fp16: bool,
    snapshot_dir: Path | None,
    window_batch_size: int,
    query_batch_size: int,
    window_max_length: int,
    query_max_length: int,
) -> dict[str, Any]:
    if not bundle_tar.is_file():
        raise ColabEncodingError(
            f"bundle not found: {bundle_tar}"
        )

    bundle_dir = work_dir / "bundle"
    output_dir = work_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] extract bundle: {bundle_tar}")
    bundle_manifest = _extract_bundle(bundle_tar, bundle_dir)
    dataset_id = str(bundle_manifest["dataset_id"])

    windows = _read_jsonl(bundle_dir / BUNDLE_WINDOWS)
    queries = _read_jsonl(bundle_dir / BUNDLE_QUERIES)
    print(f"      windows={len(windows)} queries={len(queries)}")

    print("[2/4] load encoder")
    args = argparse.Namespace(
        dry_run=dry_run,
        model_id=model_id,
        model_revision=model_revision,
        device=device,
        use_fp16=use_fp16,
        snapshot_dir=snapshot_dir,
    )
    encoder, encoder_kind = _build_encoder(args)

    print("[3/4] encode windows")
    window_texts = [str(w["text"]) for w in windows]
    window_vectors = _encode_in_batches(
        encoder,
        window_texts,
        batch_size=window_batch_size,
        max_length=window_max_length,
        label="windows",
    )
    window_rows = [
        {"doc_id": str(w["doc_id"]), "dense_vector": vec}
        for w, vec in zip(windows, window_vectors, strict=True)
    ]
    n_window = _write_jsonl_atomic(
        output_dir / OUTPUT_WINDOW_EMBEDDINGS, window_rows
    )

    print("[4/4] encode queries")
    query_texts = [str(q["text"]) for q in queries]
    query_vectors = _encode_in_batches(
        encoder,
        query_texts,
        batch_size=query_batch_size,
        max_length=query_max_length,
        label="queries",
    )
    query_rows = [
        {"query_id": str(q["query_id"]), "dense_vector": vec}
        for q, vec in zip(queries, query_vectors, strict=True)
    ]
    n_query = _write_jsonl_atomic(
        output_dir / OUTPUT_QUERY_EMBEDDINGS, query_rows
    )

    window_sha = _sha256_file(
        output_dir / OUTPUT_WINDOW_EMBEDDINGS
    )
    query_sha = _sha256_file(
        output_dir / OUTPUT_QUERY_EMBEDDINGS
    )

    manifest = {
        "artifact_type": ENCODING_ARTIFACT_TYPE,
        "artifact_version": ENCODING_ARTIFACT_VERSION,
        "dataset_id": dataset_id,
        "encoder_kind": encoder_kind,
        "model_id": model_id,
        "model_revision": model_revision,
        "dense_dimension": DENSE_DIMENSION,
        "dense_normalized": True,
        "use_fp16": bool(use_fp16 and device == "cuda"),
        "device": device,
        "window_count": n_window,
        "query_count": n_query,
        "window_embeddings_sha256": window_sha,
        "query_embeddings_sha256": query_sha,
        "source_bundle_sha256": _sha256_file(bundle_tar),
    }
    _write_json(output_dir / OUTPUT_MANIFEST, manifest)

    checksums = {
        OUTPUT_WINDOW_EMBEDDINGS: window_sha,
        OUTPUT_QUERY_EMBEDDINGS: query_sha,
        OUTPUT_MANIFEST: _sha256_file(output_dir / OUTPUT_MANIFEST),
    }
    _write_json(output_dir / OUTPUT_CHECKSUMS, checksums)

    _pack_output(output_dir, output_tar)
    manifest["output_tar_sha256"] = _sha256_file(output_tar)
    manifest["output_tar_bytes"] = output_tar.stat().st_size
    return manifest


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode a Metadata-RAG bundle with BGE-M3 dense-only and "
            "package the resulting embeddings."
        )
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("_metadata_rag_encode_work"),
    )
    parser.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    parser.add_argument(
        "--model-revision", default=MODEL_REVISION_DEFAULT
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Optional local snapshot path for offline encoding.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--use-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--window-batch-size", type=int, default=2)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--window-max-length", type=int, default=8192)
    parser.add_argument("--query-max-length", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        default=False,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    print(f"bundle:   {args.bundle}")
    print(f"output:   {args.output}")
    print(f"device:   {args.device}")
    print(f"dry_run:  {args.dry_run}")

    manifest = run_encode(
        bundle_tar=args.bundle,
        output_tar=args.output,
        work_dir=args.work_dir,
        dry_run=args.dry_run,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        use_fp16=args.use_fp16,
        snapshot_dir=args.snapshot_dir,
        window_batch_size=args.window_batch_size,
        query_batch_size=args.query_batch_size,
        window_max_length=args.window_max_length,
        query_max_length=args.query_max_length,
    )

    print("[OK] encoding complete")
    print(f"  dataset_id     = {manifest['dataset_id']}")
    print(f"  encoder_kind   = {manifest['encoder_kind']}")
    print(f"  window_count   = {manifest['window_count']}")
    print(f"  query_count    = {manifest['query_count']}")
    print(f"  output_bytes   = {manifest['output_tar_bytes']}")
    print(f"  output_sha256  = {manifest['output_tar_sha256']}")

    if not args.keep_work_dir:
        shutil.rmtree(args.work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
