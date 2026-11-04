"""CLI: build chunks for (variant, dataset) pairs. GT-aware sampling."""
import argparse
import time

from ablation.chunks.builder import build_chunks_for
from ablation.orchestration.planner import build_plan


KNOWN = {"A4_full_sah", "A7_paragraph_parents", "A1_flat_hybrid",
         "A9_placebo_boundaries", "A6_boundaries_disabled"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--tier", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sampling", action="store_true",
                    help="Disable GT-aware sampling (full corpus)")
    args = ap.parse_args(argv)

    jobs = build_plan()
    if args.tier is not None:
        jobs = [j for j in jobs if j["tier"] == args.tier]
    seen = set()
    unique = []
    for j in jobs:
        if j["variant"] not in KNOWN:
            continue
        key = (j["variant"], j["dataset"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(j)
    if args.variant:
        unique = [j for j in unique if j["variant"] == args.variant]
    if args.dataset:
        unique = [j for j in unique if j["dataset"] == args.dataset]

    print(f"Building {len(unique)} (variant, dataset) chunk sets")
    print(f"Sampling: {'DISABLED' if args.no_sampling else 'GT-aware (enabled)'}")
    print()

    counts = {}
    t0 = time.time()
    for i, j in enumerate(unique, 1):
        res = build_chunks_for(j["variant"], j["dataset"],
                               force=args.force,
                               apply_sampling=not args.no_sampling)
        s = res.get("status", "?")
        counts[s] = counts.get(s, 0) + 1
        marker = {"done": "+", "already_done": "=", "skipped": "-", "error": "!"}.get(s, "?")
        extra = ""
        if s == "done":
            extra = f"{res['n_docs']:>6} docs ({res['n_docs_skipped']:>6} skipped), {res['n_chunks']:>7} chunks, {res['size_mb']:>6.1f} MB"
        elif s == "already_done":
            extra = res["path"]
        elif s == "error":
            extra = res.get("reason", "")
        print(f"[{i:>3}/{len(unique)}] {marker} {j['variant']:<24} {j['dataset']:<20}  {extra}")

    print()
    print(f"Done in {time.time()-t0:.1f}s: {counts}")
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
