"""Evaluate a TREC run file against Pyserini qrels using pytrec_eval.

Windows-safe alternative to `python -m pyserini.eval.trec_eval`.

Usage:
    python eval_ndcg.py <run_file> <topics_id> [metric] [cutoff]

- metric : trec_eval measure name; dots or underscores both accepted
           (e.g. "ndcg_cut.10", "ndcg_cut_10", "recip_rank")
- cutoff : optional; keep only top-N docs per query before evaluation

Examples:
    # BEIR SciFact (nDCG@10)
    python eval_ndcg.py runs/scifact.bm25.trec beir-v1.0.0-scifact-test ndcg_cut.10

    # MS MARCO passage v1 (MRR@10: truncate to 10, then recip_rank)
    python eval_ndcg.py runs/msmarco_passage_v1.bm25.trec ^
        msmarco-passage-dev-subset recip_rank 10
"""
import sys
from pathlib import Path

import pytrec_eval
from pyserini.search import get_qrels_file


def normalize_metric(name: str) -> str:
    return name.replace(".", "_")


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def load_trec_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, _, score, _ = parts
            run.setdefault(qid, {})[docid] = float(score)
    return run


def truncate_run(run, cutoff: int):
    return {
        qid: dict(sorted(docs.items(), key=lambda x: -x[1])[:cutoff])
        for qid, docs in run.items()
    }


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    run_path = Path(sys.argv[1]).expanduser().resolve()
    topics_id = sys.argv[2]
    metric_in = sys.argv[3] if len(sys.argv) > 3 else "ndcg_cut.10"
    cutoff = int(sys.argv[4]) if len(sys.argv) > 4 else None
    metric = normalize_metric(metric_in)

    if not run_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    qrels_path = Path(get_qrels_file(topics_id))
    qrels = load_qrels(qrels_path)
    run = load_trec_run(run_path)

    if cutoff is not None:
        run = truncate_run(run, cutoff)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric})
    scores = evaluator.evaluate(run)

    if not scores:
        raise RuntimeError("Evaluator returned no scores.")

    sample = next(iter(scores.values()))
    if metric not in sample:
        raise KeyError(f"{metric!r} not in output. Available: {sorted(sample)}")

    values = [s[metric] for s in scores.values()]
    mean = sum(values) / len(values)

    print(f"metric:  {metric}")
    print(f"cutoff:  {cutoff if cutoff else 'none'}")
    print(f"queries: {len(values)}")
    print(f"mean:    {mean:.6f}")
    print(f"qrels:   {qrels_path}")
    print(f"run:     {run_path}")


if __name__ == "__main__":
    main()