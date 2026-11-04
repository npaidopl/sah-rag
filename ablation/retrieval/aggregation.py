"""Child → parent aggregation strategies (B-group ablations)."""
from collections import defaultdict


def aggregate_to_parents(
    ranked_chunks: list[tuple[int, float]],
    chunk_to_parent: list[str],
    strategy: str = "best_plus_margin",
    margin: float = 0.05,
) -> list[tuple[str, float]]:
    """Group chunks by parent, compute parent score, return sorted [(parent_id, score)]."""
    if not ranked_chunks:
        return []
    by_parent: dict[str, list[float]] = defaultdict(list)
    for idx, score in ranked_chunks:
        pid = chunk_to_parent[idx]
        by_parent[pid].append(float(score))

    parent_scores: dict[str, float] = {}
    for pid, scores in by_parent.items():
        scores_sorted = sorted(scores, reverse=True)
        best = scores_sorted[0]
        rest = scores_sorted[1:]
        n = len(scores_sorted)

        if strategy == "best_plus_margin":
            parent_scores[pid] = best + margin * sum(rest)
        elif strategy == "best_only":
            parent_scores[pid] = best
        elif strategy == "sum":
            parent_scores[pid] = sum(scores_sorted)
        elif strategy == "mean":
            parent_scores[pid] = sum(scores_sorted) / n
        elif strategy == "size_controlled":
            # B7: mean of rest instead of sum, to eliminate size bias
            parent_scores[pid] = best + margin * (sum(rest) / len(rest) if rest else 0.0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    out = sorted(parent_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return out


def resolve_to_documents(
    ranked_parents: list[tuple[str, float]],
    parent_to_doc: dict[str, str],
) -> list[tuple[str, float, str]]:
    """Document score = max over its parents. Returns [(doc_id, score, best_parent_id)]."""
    by_doc: dict[str, tuple[float, str]] = {}
    for pid, score in ranked_parents:
        did = parent_to_doc[pid]
        if did not in by_doc or score > by_doc[did][0]:
            by_doc[did] = (score, pid)
    return sorted(((did, s, pid) for did, (s, pid) in by_doc.items()),
                  key=lambda t: (-t[1], t[0]))