"""Master orchestrator: run Optimal Operation + Ablation + Controlled + Native E2E.

Usage:
    python -m orchestration.run_all --manifest configs/run_manifest_full.yaml
    python -m orchestration.run_all --manifest configs/run_manifest_full.yaml --dry-run
    python -m orchestration.run_all --manifest configs/run_manifest_full.yaml --resume
    python -m orchestration.run_all --manifest configs/run_manifest_full.yaml --stage 3

Stages:
    0: preflight    (env check, model check, dataset check)
    1: optimal      (Optimal Operation, subprocess to run_full_pipeline.py)
    2: ablation     (Ablation study, subprocess to ablation/run_all.py)
    3: controlled   (Controlled adaptation, in-process via scheduler)
    4: native       (Native E2E, in-process via scheduler)
    5: reporting    (Aggregate + write reports)

State persistence: output_root/run_all_state.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# =====================================================================
# Environment loading (must happen before any getenv call)
# =====================================================================


def _ensure_env_loaded() -> None:
    """Load .env from project root once if RAG_CANONICAL_ROOT is missing."""
    if os.getenv("RAG_CANONICAL_ROOT"):
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



# =====================================================================
# State management
# =====================================================================


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageState:
    done: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float = 0.0
    error: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class RunAllState:
    version: str = "1.0"
    started_at: str = ""
    finished_at: str | None = None
    stages: dict = field(default_factory=dict)

    def get_stage(self, stage_id: str) -> StageState:
        if stage_id not in self.stages:
            return StageState()
        raw = self.stages[stage_id]
        return StageState(**raw)

    def set_stage(self, stage_id: str, stage: StageState) -> None:
        self.stages[stage_id] = asdict(stage)


def load_or_create_state(path: Path) -> RunAllState:
    if not path.exists():
        return RunAllState(started_at=_now_utc())
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    state = RunAllState(
        version=raw.get("version", "1.0"),
        started_at=raw.get("started_at", _now_utc()),
        finished_at=raw.get("finished_at"),
        stages=raw.get("stages", {}),
    )
    return state


def save_state(state: RunAllState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)
    tmp.replace(path)


# =====================================================================
# Master manifest
# =====================================================================


def load_master_manifest(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_output_root(manifest: dict) -> Path:
    raw = manifest.get("global", {}).get("output_root")
    if not raw:
        raise RuntimeError("global.output_root is required in master manifest.")
    return Path(raw)


# =====================================================================
# Stage 0: Preflight
# =====================================================================


def _check_env_var(name: str, required: bool = True) -> tuple[bool, str]:
    val = os.getenv(name)
    if not val:
        return (not required, f"{name}: NOT SET" + (" (REQUIRED)" if required else " (optional)"))
    p = Path(val)
    if name.endswith("_ROOT") and not p.exists():
        return (False, f"{name}: {val} (does not exist)")
    return (True, f"{name}: {val}")


def _check_model_dir(
    env_var: str,
    expected_size_gb: float,
    default_path: str | None = None,
) -> tuple[bool, str]:
    """Check a model dir via env var, with optional fallback to a default path.

    If env var is unset AND default_path is None -> FAIL (required).
    If env var is unset AND default_path is set  -> fall back to default (optional).
    """
    val = os.getenv(env_var)
    source = env_var
    if not val:
        if default_path is None:
            return (False, f"{env_var}: NOT SET (required)")
        val = default_path
        source = f"{env_var} (unset) -> fallback to {default_path}"

    p = Path(val)
    if not p.exists():
        return (False, f"{source}: {val} (does not exist)")
    size_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    size_gb = size_bytes / (1024 ** 3)
    ok = size_gb >= expected_size_gb * 0.9
    return (ok, f"{source}: {size_gb:.2f} GB (expected >={expected_size_gb} GB)")


def _preflight_as_stage(manifest: dict, state: RunAllState) -> StageState:
    """Wrap stage_00_preflight as a StageState for uniform main() handling.

    Returns:
        StageState with done=True if preflight passed, else error set (fatal).
    """
    st = StageState(started_at=_now_utc())
    try:
        passed = stage_00_preflight(manifest, state)
        st.done = passed
        if not passed:
            st.error = "preflight checks failed (see log above)"
    except Exception as exc:
        st.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    st.finished_at = _now_utc()
    return st


def stage_00_preflight(manifest: dict, state: RunAllState) -> bool:
    print("\n" + "=" * 72)
    print("  STAGE 0: PREFLIGHT")
    print("=" * 72)

    cfg = manifest.get("preflight", {})
    checks_passed = True

    # 1. Env vars
    print("\n[1/4] Environment variables...")
    required_env = cfg.get("check_env", ["RAG_CANONICAL_ROOT", "RAG_INDOOR_LLM_ROOT"])
    for name in required_env:
        ok, msg = _check_env_var(name, required=True)
        print(f"  {'[OK]' if ok else '[FAIL]':8s} {msg}")
        if not ok:
            checks_passed = False

    # 2. Models
    print("\n[2/4] Model directories...")
    model_checks = cfg.get("check_models", [])
    # model_env_map: (env_var, min_size_gb, default_path_or_None)
    model_env_map = {
        "Qwen/Qwen2.5-14B-Instruct": (
            "RAG_INDOOR_LLM_ROOT", 25.0, None,
        ),
        "Qwen/Qwen2.5-32B-Instruct": (
            "RAG_GENERATION_LLM_ROOT", 55.0, None,
        ),
        "BAAI/bge-m3": (
            "RAG_EMBEDDER_ROOT", 3.5,
            "D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/"
            "5617a9f61b028005a4858fdac845db406aefb181",
        ),
    }
    for model_name in model_checks:
        if model_name in model_env_map:
            env_var, min_size, default_path = model_env_map[model_name]
            ok, msg = _check_model_dir(env_var, min_size, default_path=default_path)
            print(f"  {'[OK]' if ok else '[FAIL]':8s} {model_name}: {msg}")
            if not ok and default_path is None:
                checks_passed = False

    # 3. Datasets
    print("\n[3/4] Datasets...")
    canonical_root = Path(os.getenv("RAG_CANONICAL_ROOT", ""))
    required_ds = cfg.get("check_datasets", {}).get("required_ids", [])
    if canonical_root.exists():
        found = [d.name for d in canonical_root.iterdir() if d.is_dir()]
        for ds in required_ds:
            ok = ds in found
            print(f"  {'[OK]' if ok else '[FAIL]':8s} {ds}")
            if not ok:
                checks_passed = False
        print(f"\n  Total: {len(found)} dirs (expected {len(required_ds)})")
    else:
        print(f"  [FAIL] RAG_CANONICAL_ROOT does not exist: {canonical_root}")
        checks_passed = False

    # 4. Optional workspace for Optimal Operation
    print("\n[4/4] Optimal Operation workspace (optional)...")
    ws = os.getenv("SAH_RAG_OPTIMAL_WORKSPACE")
    if ws and Path(ws).exists():
        print(f"  [OK]     SAH_RAG_OPTIMAL_WORKSPACE: {ws}")
    else:
        print(f"  [SKIP]   SAH_RAG_OPTIMAL_WORKSPACE not set or not found")
        print(f"           Stage 1 will be skipped (graceful)")

    return checks_passed


# =====================================================================
# Stage 1: Optimal Operation (subprocess)
# =====================================================================


def stage_01_optimal(manifest: dict, state: RunAllState) -> StageState:
    print("\n" + "=" * 72)
    print("  STAGE 1: OPTIMAL OPERATION")
    print("=" * 72)

    st = StageState(started_at=_now_utc())

    if not manifest.get("optimal_operation", {}).get("enabled", False):
        print("  [SKIP] Stage disabled in manifest")
        st.done = True
        return st

    ws = os.getenv("SAH_RAG_OPTIMAL_WORKSPACE")
    if not ws or not Path(ws).exists():
        print(f"  [SKIP] SAH_RAG_OPTIMAL_WORKSPACE not set or not found")
        print(f"         Set env var to enable Optimal Operation stage")
        st.done = True
        st.extra["skip_reason"] = "workspace_not_set"
        return st

    pipeline_script = Path(ws) / "scripts" / "run_full_pipeline.py"
    if not pipeline_script.exists():
        print(f"  [FAIL] Pipeline script not found: {pipeline_script}")
        st.error = f"pipeline script missing: {pipeline_script}"
        return st

    print(f"  Workspace: {ws}")
    print(f"  Pipeline: {pipeline_script}")

    import time
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(pipeline_script), "--skip-gpu"],
            cwd=str(ws),
            check=False,
        )
        st.duration_s = time.time() - t0
        if result.returncode == 0:
            st.done = True
            print(f"  [OK] Optimal Operation completed in {st.duration_s:.1f}s")
        else:
            st.error = f"pipeline exited with code {result.returncode}"
            print(f"  [FAIL] {st.error}")
    except Exception as exc:
        st.duration_s = time.time() - t0
        st.error = f"{type(exc).__name__}: {exc}"
        print(f"  [FAIL] {st.error}")
        traceback.print_exc()

    st.finished_at = _now_utc()
    return st


# =====================================================================
# Stage 2: Ablation (subprocess)
# =====================================================================


def stage_02_ablation(manifest: dict, state: RunAllState) -> StageState:
    print("\n" + "=" * 72)
    print("  STAGE 2: ABLATION STUDY")
    print("=" * 72)

    st = StageState(started_at=_now_utc())

    if not manifest.get("ablation", {}).get("enabled", False):
        print("  [SKIP] Stage disabled in manifest")
        st.done = True
        return st

    ablation_script = _PROJECT_ROOT / "ablation" / "run_all.py"
    if not ablation_script.exists():
        print(f"  [FAIL] Ablation script not found: {ablation_script}")
        st.error = f"ablation script missing: {ablation_script}"
        return st

    args = [sys.executable, str(ablation_script)]
    tier = manifest.get("ablation", {}).get("tier")
    if tier:
        args.extend(["--tier", str(tier)])

    print(f"  Command: {' '.join(args)}")
    import time
    t0 = time.time()
    try:
        result = subprocess.run(args, cwd=str(_PROJECT_ROOT), check=False)
        st.duration_s = time.time() - t0
        if result.returncode == 0:
            st.done = True
            print(f"  [OK] Ablation completed in {st.duration_s:.1f}s")
        else:
            st.error = f"ablation exited with code {result.returncode}"
            print(f"  [FAIL] {st.error}")
    except Exception as exc:
        st.duration_s = time.time() - t0
        st.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    st.finished_at = _now_utc()
    return st


# =====================================================================
# Stage 3: Controlled Adaptation (in-process)
# =====================================================================


def _build_scheduler_manifest(
    *,
    modes: list[str],
    baselines: list[str],
    datasets: list[str],
    top_k: int,
    output_root: Path,
    manifest_path: Path,
) -> None:
    data = {
        "modes": modes,
        "baselines": baselines,
        "datasets": datasets,
        "top_k": top_k,
        "output_root": str(output_root),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _run_scheduler_stage(
    stage_name: str,
    *,
    modes: list[str],
    baselines: list[str],
    datasets: list[str],
    top_k: int,
    output_root: Path,
    state: RunAllState,
    dry_run: bool = False,
) -> StageState:
    print(f"\n" + "=" * 72)
    print(f"  {stage_name}")
    print("=" * 72)

    st = StageState(started_at=_now_utc())

    if not modes or not baselines or not datasets:
        print(f"  [SKIP] Empty modes/baselines/datasets")
        st.done = True
        return st

    manifest_path = output_root / f"_generated_{stage_name.split(':')[0].strip().lower().replace(' ', '_')}.yaml"
    _build_scheduler_manifest(
        modes=modes,
        baselines=baselines,
        datasets=datasets,
        top_k=top_k,
        output_root=output_root,
        manifest_path=manifest_path,
    )
    print(f"  Manifest: {manifest_path}")
    print(f"  Modes: {modes}")
    print(f"  Baselines ({len(baselines)}): {baselines}")
    print(f"  Datasets ({len(datasets)}): {datasets}")
    print(f"  Total cells: {len(modes) * len(baselines) * len(datasets)}")

    from orchestration.scheduler import run_manifest

    import time
    t0 = time.time()
    try:
        counts = run_manifest(manifest_path, dry_run=dry_run)
        st.duration_s = time.time() - t0
        st.extra["counts"] = counts
        if counts.get("failed", 0) == 0:
            st.done = True
            print(f"  [OK] {stage_name} complete in {st.duration_s:.1f}s")
            print(f"       done={counts['done']} skipped={counts['skipped']} failed={counts['failed']}")
        else:
            st.error = f"{counts['failed']} cells failed"
            print(f"  [FAIL] {st.error} (see {output_root} for error.log)")
    except Exception as exc:
        st.duration_s = time.time() - t0
        st.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    st.finished_at = _now_utc()
    return st


def stage_03_controlled(manifest: dict, state: RunAllState, dry_run: bool = False) -> StageState:
    cfg = manifest.get("controlled_adaptation", {})
    if not cfg.get("enabled", False):
        print("\n" + "=" * 72)
        print("  STAGE 3: CONTROLLED ADAPTATION [SKIP - disabled in manifest]")
        print("=" * 72)
        st = StageState(started_at=_now_utc(), done=True)
        return st

    output_root = get_output_root(manifest) / "controlled_adaptation"
    return _run_scheduler_stage(
        "STAGE 3: CONTROLLED ADAPTATION",
        modes=["controlled"],
        baselines=cfg.get("baselines", []),
        datasets=cfg.get("datasets", []),
        top_k=cfg.get("top_k", 10),
        output_root=output_root,
        state=state,
        dry_run=dry_run,
    )


# =====================================================================
# Stage 4: Native E2E (in-process)
# =====================================================================


def stage_04_native(manifest: dict, state: RunAllState, dry_run: bool = False) -> StageState:
    cfg = manifest.get("native_e2e", {})
    if not cfg.get("enabled", False):
        print("\n" + "=" * 72)
        print("  STAGE 4: NATIVE E2E [SKIP - disabled in manifest]")
        print("=" * 72)
        st = StageState(started_at=_now_utc(), done=True)
        return st

    output_root = get_output_root(manifest) / "native_e2e"
    return _run_scheduler_stage(
        "STAGE 4: NATIVE E2E",
        modes=["native"],
        baselines=cfg.get("baselines", []),
        datasets=cfg.get("datasets", []),
        top_k=cfg.get("top_k", 10),
        output_root=output_root,
        state=state,
        dry_run=dry_run,
    )


# =====================================================================
# Stage 5: Reporting
# =====================================================================


def stage_05_reporting(manifest: dict, state: RunAllState) -> StageState:
    print("\n" + "=" * 72)
    print("  STAGE 5: REPORTING")
    print("=" * 72)

    st = StageState(started_at=_now_utc())

    if not manifest.get("reporting", {}).get("enabled", False):
        print("  [SKIP] Stage disabled")
        st.done = True
        return st

    output_root = get_output_root(manifest)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate all metrics.json files
    all_metrics = []
    for mfile in output_root.rglob("metrics.json"):
        try:
            with mfile.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data["_source"] = str(mfile.relative_to(output_root))
            all_metrics.append(data)
        except Exception as e:
            print(f"  [WARN] Could not read {mfile}: {e}")

    # Aggregate summary
    summary = {
        "generated_at": _now_utc(),
        "total_runs": len(all_metrics),
        "runs_by_baseline": {},
        "runs_by_dataset": {},
        "runs_by_mode": {},
        "metrics_aggregate": {},
    }
    for m in all_metrics:
        b = m.get("baseline", "?")
        d = m.get("dataset", "?")
        mode = m.get("mode", "?")
        summary["runs_by_baseline"][b] = summary["runs_by_baseline"].get(b, 0) + 1
        summary["runs_by_dataset"][d] = summary["runs_by_dataset"].get(d, 0) + 1
        summary["runs_by_mode"][mode] = summary["runs_by_mode"].get(mode, 0) + 1

    with (reports_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Closure manifest
    closure = {
        "generated_at": _now_utc(),
        "state": asdict(state),
        "manifest": manifest,
        "total_metrics_files": len(all_metrics),
    }
    with (reports_dir / "closure_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(closure, f, indent=2)

    print(f"  [OK] Summary: {reports_dir / 'summary.json'}")
    print(f"       Total runs: {len(all_metrics)}")
    print(f"       By baseline: {summary['runs_by_baseline']}")
    print(f"       By dataset: {summary['runs_by_dataset']}")

    st.done = True
    st.finished_at = _now_utc()
    st.extra["total_runs"] = len(all_metrics)
    return st


# =====================================================================
# Main entry point
# =====================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orchestration.run_all")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Master manifest YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from state.json")
    parser.add_argument("--stage", type=int, default=None,
                        help="Run only a specific stage (0-5)")
    parser.add_argument("--force-stage", action="store_true",
                        help="Re-run even if state says done")
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        print(f"[FAIL] Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    manifest = load_master_manifest(args.manifest)
    output_root = get_output_root(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "run_all_state.json"
    state = load_or_create_state(state_path) if args.resume else RunAllState(started_at=_now_utc())

    print("=" * 72)
    print(f"  SAH-RAG run_all orchestrator")
    print(f"  Manifest: {args.manifest}")
    print(f"  Output:   {output_root}")
    print(f"  State:    {state_path}")
    print(f"  Resume:   {args.resume}")
    print("=" * 72)

    stages = [
        ("0_preflight",   lambda: _preflight_as_stage(manifest, state)),
        ("1_optimal",     lambda: stage_01_optimal(manifest, state)),
        ("2_ablation",    lambda: stage_02_ablation(manifest, state)),
        ("3_controlled",  lambda: stage_03_controlled(manifest, state, dry_run=args.dry_run)),
        ("4_native",      lambda: stage_04_native(manifest, state, dry_run=args.dry_run)),
        ("5_reporting",   lambda: stage_05_reporting(manifest, state)),
    ]

    # If --stage specified, run only that stage
    if args.stage is not None:
        if not 0 <= args.stage < len(stages):
            print(f"[FAIL] --stage must be 0-{len(stages)-1}", file=sys.stderr)
            return 1
        stages = [stages[args.stage]]

    overall_exit = 0
    for stage_id, stage_fn in stages:
        prior = state.get_stage(stage_id)
        if prior.done and args.resume and not args.force_stage:
            print(f"\n[{stage_id}] SKIP (already done, use --force-stage to re-run)")
            continue

        print(f"\n[{stage_id}] START")
        try:
            result = stage_fn()
        except Exception as exc:
            print(f"\n[{stage_id}] FATAL: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            st = StageState(error=f"{type(exc).__name__}: {exc}", started_at=_now_utc(), finished_at=_now_utc())
            state.set_stage(stage_id, st)
            save_state(state, state_path)
            overall_exit = 2
            break

        # Stage returns either bool (preflight) or StageState
        if isinstance(result, bool):
            st = StageState(done=result, started_at=_now_utc(), finished_at=_now_utc())
        else:
            st = result

        state.set_stage(stage_id, st)
        save_state(state, state_path)

        if st.error:
            print(f"\n[{stage_id}] FAILED: {st.error}")
            overall_exit = 2
            if manifest.get("global", {}).get("stop_on_fatal", True):
                print(f"[{stage_id}] Stopping (stop_on_fatal=true)")
                break
        elif not st.done:
            print(f"\n[{stage_id}] NOT DONE (partial or skipped)")

    state.finished_at = _now_utc()
    save_state(state, state_path)

    print("\n" + "=" * 72)
    print(f"  run_all complete. State: {state_path}")
    print(f"  Exit code: {overall_exit}")
    print("=" * 72)

    return overall_exit


if __name__ == "__main__":
    sys.exit(main())