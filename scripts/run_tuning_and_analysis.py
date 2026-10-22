"""Run CPU tuning grid and truncation analysis."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

from sah_rag.canonical.paths import resolve_dataset_paths
from evaluation import load_clerk_retrieval_dataset


def run_grid() -> None:
    """Run benchmark for all configurations."""
    configurations = [
        {"batch": 4, "length": 256},
        {"batch": 8, "length": 256},
        {"batch": 16, "length": 256},
        {"batch": 4, "length": 384},
        {"batch": 8, "length": 384},
        {"batch": 16, "length": 384},
        {"batch": 4, "length": 512},
        {"batch": 8, "length": 512},
    ]

    print("=== RUNNING CPU TUNING GRID (50 pairs) ===")
    for cfg in configurations:
        print(f"\n--- BATCH {cfg['batch']} | LENGTH {cfg['length']} ---")
        cmd = [
            sys.executable,
            "scripts\\benchmark_bge_reranker_cpu.py",
            "--batch-size",
            str(cfg["batch"]),
            "--max-length",
            str(cfg["length"]),
            "--pair-count",
            "50",
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(
                f"Warning: Batch={cfg['batch']}, Length={cfg['length']} failed."
            )


def summarize_results() -> None:
    """Summarize tuning grid results."""
    print("\n=== TUNING GRID SUMMARY ===")
    files = sorted(Path("outputs").glob("bge_reranker_cpu_batch_*_length_*.json"))
    rows = []
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            r = d["result"]
            rows.append((
                d["batch_size"],
                d["max_length"],
                r["elapsed_seconds"],
                r["pairs_per_second"],
                r["seconds_per_pair"],
            ))
        except Exception:
            pass
    if not rows:
        print("No benchmark results found.")
        return
    rows.sort(key=lambda x: (-x[3], x[1], x[0]))
    print("BATCH | LENGTH | SECONDS | PAIRS/S | SEC/PAIR | 20K PAIRS HOURS")
    print("-" * 65)
    for b, length_val, s, rate, sec in rows:
        hours = 20000 / rate / 3600
        print(
            f"{b:5} | {length_val:6} | {s:7.2f} | {rate:7.3f} |"
            f" {sec:8.3f} | {hours:15.2f}"
        )


def percentile(values: list[int], fraction: float) -> int:
    """Return percentile value."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def run_truncation() -> None:
    """Run truncation analysis."""
    print("\n=== RUNNING TRUNCATION ANALYSIS ===")
    model_report_path = Path("outputs/bge_reranker_v2_m3_download.json")
    if not model_report_path.exists():
        print("Model report not found.")
        return
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_report["snapshot_path"],
        local_files_only=True,
        use_fast=True,
    )
    dataset = load_clerk_retrieval_dataset()
    queries = [q for q in dataset.queries if q.split == "test"][:10]
    documents_path = Path(resolve_dataset_paths("clerk").documents)
    documents = [
        json.loads(line)["text_normalized"]
        for line in documents_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    lengths: list[int] = []
    for query_index, query in enumerate(queries, start=1):
        for document in documents:
            encoded = tokenizer(
                query.query_text,
                str(document),
                add_special_tokens=True,
                truncation=False,
            )
            lengths.append(len(encoded["input_ids"]))
        print(f"[QUERY] {query_index}/{len(queries)} {query.query_id}")

    print("[OK] Pair count:", len(lengths))
    print("[OK] Minimum:", min(lengths))
    print("[OK] Median:", statistics.median(lengths))
    print("[OK] P90:", percentile(lengths, 0.90))
    print("[OK] P95:", percentile(lengths, 0.95))
    print("[OK] P99:", percentile(lengths, 0.99))
    print("[OK] Maximum:", max(lengths))

    for limit in (256, 384, 512):
        truncated = sum(length > limit for length in lengths)
        print(
            f"[LIMIT] {limit} truncated={truncated} "
            f"rate={round(truncated / len(lengths), 4)}"
        )


if __name__ == "__main__":
    run_grid()
    summarize_results()
    run_truncation()
