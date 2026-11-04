"""Entry point: python -m ablation.run_all [--plan] [--dry-run] [--mock-embeddings] ..."""
import argparse
import json
import time

from ablation.config import LOGS_DIR, ensure_dirs
from ablation.orchestration.planner import build_plan, plan_summary
from ablation.orchestration.runner import run_job


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock-embeddings", action="store_true")
    ap.add_argument("--job", nargs=2, metavar=("VARIANT", "DATASET"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tier", type=int, choices=[1, 2, 3])
    ap.add_argument("--variant", type=str)
    ap.add_argument("--dataset", type=str)
    args = ap.parse_args(argv)

    ensure_dirs()
    jobs = build_plan()
    if args.tier is not None:
        jobs = [j for j in jobs if j["tier"] == args.tier]
    if args.variant:
        jobs = [j for j in jobs if j["variant"] == args.variant]
    if args.dataset:
        jobs = [j for j in jobs if j["dataset"] == args.dataset]
    if args.job:
        v, ds = args.job
        jobs = [j for j in jobs if j["variant"] == v and j["dataset"] == ds]

    print(f"Plan: {len(jobs)} jobs")
    if args.plan:
        print(json.dumps(plan_summary(jobs), indent=2))
        for j in jobs:
            print(f"  {j['tier']}  {j['variant']:<30}  {j['dataset']}")
        return 0

    if args.limit:
        jobs = jobs[:args.limit]
    print(f"Executing: {len(jobs)} jobs "
          f"(dry_run={args.dry_run}, mock_embeddings={args.mock_embeddings})")
    print()

    log_path = LOGS_DIR / f"run_all_{int(time.time())}.jsonl"
    counts = {}
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        for i, job in enumerate(jobs, 1):
            res = run_job(job, dry_run=args.dry_run, mock_embeddings=args.mock_embeddings)
            status = res.get("status", "?")
            counts[status] = counts.get(status, 0) + 1
            log.write(json.dumps({"job": job, "result": res}) + "\n")
            log.flush()
            marker = {"done": "+", "already_done": "=", "skipped": "-",
                      "error": "!"}.get(status, "?")
            extra = res.get("reason", res.get("elapsed_s", ""))
            print(f"[{i:>3}/{len(jobs)}] {marker} {job['variant']:<28} {job['dataset']:<18}  {extra}")

    print()
    print(f"Done in {time.time()-t0:.1f}s")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"  log: {log_path}")
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())