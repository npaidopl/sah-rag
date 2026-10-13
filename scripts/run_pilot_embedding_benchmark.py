"""Bounded CPU embedding benchmark for BGE-M3 (Step 83).

Loads the exact local snapshot, runs one warmup and three measured
embedding passes per configured batch size, and writes a benchmark
report. Never persists embeddings and never touches Qdrant.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_contract.yaml"
REPORT_PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_benchmark_report.yaml"


# ---------------------------------------------------------------------------
# Contract and environment
# ---------------------------------------------------------------------------


def load_contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def apply_offline_env(contract: dict) -> None:
    for key, value in contract["offline_enforcement"].items():
        os.environ[str(key)] = str(value)


# ---------------------------------------------------------------------------
# Resource measurement
# ---------------------------------------------------------------------------


def measure_rss_bytes() -> int:
    """Return peak process working set size in bytes (Windows)."""
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    handle = kernel32.GetCurrentProcess()
    if not kernel32.K32GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    ):
        return 0
    return int(counters.PeakWorkingSetSize)

def available_ram_bytes() -> int:
    import ctypes
    from ctypes import wintypes

    class MSX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MSX()
    stat.dwLength = ctypes.sizeof(stat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullAvailPhys)


# ---------------------------------------------------------------------------
# Input and helpers
# ---------------------------------------------------------------------------


def load_child_chunks(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def normalize_sparse(weights: dict) -> tuple[list[int], list[float]]:
    items = [(int(k), float(v)) for k, v in weights.items() if float(v) > 0.0]
    items.sort(key=lambda pair: pair[0])
    indices = [i for i, _ in items]
    values = [v for _, v in items]
    return indices, values


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def select_batch_size(results: dict) -> tuple[int | None, str]:
    successful = sorted(bs for bs, r in results.items() if r.get("status") == "OK")
    if not successful:
        return None, "No batch size completed successfully."
    selected = successful[0]
    for bs in successful[1:]:
        bs_rate = results[bs]["median_records_per_second"]
        selected_rate = results[selected]["median_records_per_second"]
        if bs_rate >= selected_rate:
            selected = bs
    reason = (
        f"Selected batch size {selected}: highest tested batch size that "
        "completed without failure and had median throughput no worse than "
        "the smaller successful batch sizes."
    )
    return selected, reason


def build_report(
    contract: dict,
    environment: dict,
    chunks: list[dict],
    results: dict,
    selected: int | None,
    selection_reason: str,
    peak_rss: int,
    status: str,
    failure_stage: str | None = None,
) -> dict:
    token_counts = [int(c["token_count"]) for c in chunks]
    report = {
        "artifact_type": "pilot_embedding_benchmark_report",
        "artifact_version": "1.0",
        "benchmark": {
            "id": "bge_m3_cpu_pilot_benchmark",
            "protocol_version": "0.1.0",
            "status": status,
        },
        "environment": environment,
        "model": {
            "model_id": contract["model"]["model_id"],
            "model_revision": contract["model"]["model_revision"],
            "device": "cpu",
            "local_files_only": True,
            "snapshot_path": contract["model"]["snapshot_path"],
        },
        "input": {
            "record_count": len(chunks),
            "total_input_tokens": sum(token_counts),
            "minimum_tokens_per_record": min(token_counts),
            "median_tokens_per_record": int(median(token_counts)),
            "maximum_tokens_per_record": max(token_counts),
        },
        "results": {f"batch_size_{bs}": results[bs] for bs in sorted(results)},
        "selection": {
            "selected_batch_size": selected,
            "reason": selection_reason,
        },
        "peak_process_ram_bytes": peak_rss,
        "execution_state": {
            "model_loaded": status == "SUCCEEDED",
            "embeddings_computed_for_benchmark": status == "SUCCEEDED",
            "embeddings_persisted": False,
        },
        "model_download_performed": False,
        "network_access_performed": False,
        "qdrant_execution_performed": False,
        "retrieval_execution_performed": False,
        "evaluation_execution_performed": False,
        "generation_execution_performed": False,
        "interpretation": {
            "publishable": False,
            "real_dataset_estimate": False,
            "linear_extrapolation_allowed": False,
        },
    }
    if failure_stage is not None:
        report["failure_stage"] = failure_stage
    return report


def run_benchmark_once(model, texts: list[str], batch_size: int) -> tuple[float, int]:
    gc.collect()
    start = time.perf_counter()
    model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    elapsed = time.perf_counter() - start
    rss = measure_rss_bytes()
    return elapsed, rss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    contract = load_contract()
    apply_offline_env(contract)

    available = available_ram_bytes()
    minimum = int(contract["resource_limits"]["minimum_available_ram_bytes"])
    if available < minimum:
        raise SystemExit(
            "available RAM " + str(available) + " below minimum " + str(minimum)
        )

    chunks_path = REPO_ROOT / contract["source"]["child_chunks_path"]
    chunks = load_child_chunks(chunks_path)
    expected = int(contract["source"]["expected_child_chunk_count"])
    if len(chunks) != expected:
        raise SystemExit("expected " + str(expected) + " child chunks, found " + str(len(chunks)))

    texts = [c["text_normalized"] for c in chunks]

    env_versions = contract["model"]["environment_versions"]
    environment = {
        "python_version": env_versions["python"],
        "FlagEmbedding_version": env_versions["FlagEmbedding"],
        "torch_version": env_versions["torch"],
        "transformers_version": env_versions["transformers"],
        "huggingface_hub_version": env_versions["huggingface_hub"],
        "numpy_version": env_versions["numpy"],
        "cpu_only": True,
    }

    batch_sizes = [int(x) for x in contract["benchmark"]["batch_sizes"]]
    warmup_runs = int(contract["benchmark"]["warmup_run_count"])
    measured_runs = int(contract["benchmark"]["measured_run_count_per_batch_size"])

    peak_rss = measure_rss_bytes()
    results: dict[int, dict] = {}
    status = "SUCCEEDED"
    failure_stage: str | None = None
    selected: int | None = None
    selection_reason = ""

    try:
        from FlagEmbedding import BGEM3FlagModel

        model = BGEM3FlagModel(
            contract["model"]["snapshot_path"],
            normalize_embeddings=True,
            use_fp16=False,
            use_bf16=False,
            devices="cpu",
            pooling_method="cls",
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        peak_rss = max(peak_rss, measure_rss_bytes())

        for _ in range(warmup_runs):
            run_benchmark_once(model, texts[:1], batch_size=1)

        for bs in batch_sizes:
            timings: list[float] = []
            peak_for_bs = 0
            try:
                for _ in range(measured_runs):
                    elapsed, rss = run_benchmark_once(model, texts, batch_size=bs)
                    timings.append(elapsed)
                    peak_for_bs = max(peak_for_bs, rss)
                    peak_rss = max(peak_rss, rss)
                med_seconds = median(timings)
                med_rps = len(texts) / med_seconds
                results[bs] = {
                    "status": "OK",
                    "raw_embedding_seconds": timings,
                    "minimum_seconds": min(timings),
                    "median_seconds": med_seconds,
                    "maximum_seconds": max(timings),
                    "median_records_per_second": med_rps,
                    "median_milliseconds_per_record": 1000.0 * med_seconds / len(texts),
                    "peak_process_ram_bytes": peak_for_bs,
                }
            except Exception as exc:  # noqa: BLE001
                results[bs] = {
                    "status": "FAILED",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }

        selected, selection_reason = select_batch_size(results)
        if selected is None:
            status = "INTERRUPTED"
            failure_stage = "all_batch_sizes_failed"
    except MemoryError as exc:
        status = "INTERRUPTED"
        failure_stage = "model_load"
        results = {}
        selection_reason = "MemoryError: " + str(exc)
    except Exception as exc:  # noqa: BLE001
        status = "INTERRUPTED"
        failure_stage = "model_load"
        results = {}
        selection_reason = type(exc).__name__ + ": " + str(exc)

    report = build_report(
        contract=contract,
        environment=environment,
        chunks=chunks,
        results=results,
        selected=selected,
        selection_reason=selection_reason,
        peak_rss=peak_rss,
        status=status,
        failure_stage=failure_stage,
    )
    REPORT_PATH.write_text(
        yaml.safe_dump(report, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("[OK] Pilot BGE-M3 CPU embedding benchmark")
    print("  model:", contract["model"]["model_id"])
    print("  revision:", contract["model"]["model_revision"])
    print("  device: cpu")
    print("  input chunks:", len(chunks))
    print("  total input tokens:", report["input"]["total_input_tokens"])
    for bs in batch_sizes:
        entry = results.get(bs, {})
        if entry.get("status") == "OK":
            print("  batch size " + str(bs) + ":")
            print("    runs:", entry["raw_embedding_seconds"])
            print("    median seconds:", entry["median_seconds"])
            print("    records per second:", entry["median_records_per_second"])
        else:
            print("  batch size " + str(bs) + ": FAILED (" + str(entry.get("failure_type")) + ")")
    print("  selected batch size:", selected)
    print("  peak RSS bytes:", peak_rss)
    print("  model download performed: False")
    print("  network access performed: False")
    print("  Qdrant executed: False")
    print("  retrieval executed: False")
    print("  evaluation executed: False")
    print("  generation executed: False")
    return 0 if status == "SUCCEEDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())