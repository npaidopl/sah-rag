"""Benchmark BGE-M3 dense encoding of source-corrected long units.

This script performs model inference for resource validation only.

It does not execute foreign LongRAG code, create production embedding
artifacts, ingest Qdrant collections, or perform retrieval.
"""

from __future__ import annotations

import argparse
import gc
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer

MODEL_REPORT_PATH = Path("outputs/bge_m3_model_download.json")
UNITS_ROOT = Path("outputs/longrag_clerk_source_corrected_pilot")
OUTPUT_ROOT = Path("outputs/longrag_bge_m3_cpu_microbenchmark")
REPORT_PATH = OUTPUT_ROOT / "report.json"

CONDITIONS: dict[str, dict[str, Any]] = {
    "LONGRAG-SD-4096": {
        "target_token_size": 4096,
        "sample_size": 16,
        "batch_sizes": (8, 4, 2, 1),
        "minimum_model_input_tokens": 1024,
    },
    "LONGRAG-ADJ-2048": {
        "target_token_size": 2048,
        "sample_size": 8,
        "batch_sizes": (4, 2, 1),
        "minimum_model_input_tokens": None,
    },
    "LONGRAG-ADJ-4096": {
        "target_token_size": 4096,
        "sample_size": 4,
        "batch_sizes": (2, 1),
        "minimum_model_input_tokens": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CPU-only BGE-M3 micro-benchmark for source-corrected LongRAG units."
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(CONDITIONS.keys()),
        help="Target LongRAG condition to benchmark.",
    )
    return parser.parse_args()


def track_peak_memory(stop_event: threading.Event, peak_storage: list[float]) -> None:
    process = psutil.Process()
    while not stop_event.is_set():
        try:
            rss = process.memory_info().rss / (1024**3)
            if rss > peak_storage[0]:
                peak_storage[0] = rss
        except Exception:
            pass
        time.sleep(0.05)


def main() -> None:
    args = parse_args()
    condition_id = args.condition
    cfg = CONDITIONS[condition_id]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model_report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
    snapshot_path = Path(model_report["snapshot_path"])

    print(f"[INFO] Loading tokenizer from {snapshot_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), local_files_only=True)
    special_tokens_count = tokenizer.num_special_tokens_to_add(pair=False)
    print(f"[INFO] Tokenizer special tokens count: {special_tokens_count}")

    units_file = UNITS_ROOT / condition_id / "units.jsonl"
    if not units_file.is_file():
        raise FileNotFoundError(f"Units file not found: {units_file}")

    all_units = []
    with units_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_units.append(json.loads(line))

    sample_size = min(int(cfg["sample_size"]), len(all_units))
    sampled_units = all_units[:sample_size]
    texts = [u["text"] for u in sampled_units]

    raw_token_counts = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    min_raw = min(raw_token_counts) if raw_token_counts else 0
    max_raw = max(raw_token_counts) if raw_token_counts else 0
    print(
        f"[INFO] Sampled {sample_size} units for {condition_id}. "
        f"Raw token range: [{min_raw}, {max_raw}]"
    )

    target_size = int(cfg["target_token_size"])
    min_input_tokens = cfg["minimum_model_input_tokens"]
    if min_input_tokens is not None:
        max_length = max(min_input_tokens, target_size + special_tokens_count)
    else:
        max_length = target_size + special_tokens_count

    print(f"[INFO] Initializing BGEM3FlagModel on CPU from {snapshot_path}")
    load_started = time.perf_counter()
    model = BGEM3FlagModel(str(snapshot_path), use_fp16=False, device="cpu")
    load_seconds = time.perf_counter() - load_started
    print(f"[INFO] Model loaded in {load_seconds:.2f}s")

    benchmarks = []
    all_untruncated = True
    successful_batch_sizes = []
    recommended_batch_size = 1
    truncated_count = 0

    for batch_size in cfg["batch_sizes"]:
        print(f"[INFO] Benchmarking batch_size={batch_size} with max_length={max_length}...")
        gc.collect()

        peak_storage = [psutil.Process().memory_info().rss / (1024**3)]
        stop_event = threading.Event()
        mem_thread = threading.Thread(target=track_peak_memory, args=(stop_event, peak_storage))
        mem_thread.start()

        started = time.perf_counter()
        status = "success"
        error_message = None
        dense_vectors = []
        non_finite_count = 0
        dimension = 1024
        normalized = True

        try:
            output = model.encode(
                texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense_vectors = output["dense_vecs"]

            for text in texts:
                full_tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
                if len(full_tokens) > max_length:
                    truncated_count += 1

            for vec in dense_vectors:
                arr = np.array(vec, dtype=np.float32)
                if not np.isfinite(arr).all():
                    non_finite_count += int(np.isnan(arr).sum() + np.isinf(arr).sum())
                dimension = int(arr.shape[0])
                norm = np.linalg.norm(arr)
                if abs(norm - 1.0) > 1e-3:
                    normalized = False

        except Exception as e:
            status = "failed"
            error_message = str(e)
            all_untruncated = False

        elapsed = time.perf_counter() - started
        stop_event.set()
        mem_thread.join()

        peak_rss_gb = peak_storage[0]
        total_tokens = sum(raw_token_counts)
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

        if status == "success" and truncated_count == 0 and non_finite_count == 0:
            successful_batch_sizes.append(batch_size)
            recommended_batch_size = batch_size

        benchmarks.append({
            "batch_size": batch_size,
            "status": status,
            "error_message": error_message,
            "elapsed_seconds": round(elapsed, 3),
            "peak_rss_gb": round(peak_rss_gb, 3),
            "raw_tokens_per_second": round(tokens_per_sec, 2),
            "dense_dimension": dimension,
            "dense_normalized": normalized,
            "non_finite_value_count": non_finite_count,
            "truncated_sample_count": truncated_count,
        })

    report = {
        "conditions": [
            {
                "condition_id": condition_id,
                "sample_size": sample_size,
                "target_token_size": target_size,
                "special_tokens_count": special_tokens_count,
                "maximum_model_input_tokens": max_length,
                "raw_token_range": [min_raw, max_raw],
                "all_samples_untruncated": (all_untruncated and truncated_count == 0),
                "successful_batch_sizes": successful_batch_sizes,
                "recommended_batch_size": recommended_batch_size,
                "benchmarks": benchmarks,
            }
        ]
    }

    payload = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    REPORT_PATH.write_text(payload, encoding="utf-8")
    print(f"[OK] Benchmark report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
