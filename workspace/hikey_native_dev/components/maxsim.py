"""MaxSim section scoring (paper Section 3.2).

S_sec(s, q) = max_{c in C_s} s(c, q)

Type-specific unit score:
    Text:        s_hybrid(text(c), q)
    Table/Image: gamma * s_vis(c, q) + (1 - gamma) * s_hybrid(ctx(c), q)

Final section score:
    S_final(s, q) = lambda * S_doc(doc(s), q) + (1 - lambda) * S_sec(s, q)
"""
from __future__ import annotations

import math
from typing import Callable


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def maxsim_section_score(
    query_emb: list[float],
    unit_embs: list[list[float]],
) -> float:
    if not unit_embs:
        return 0.0
    return max(cosine(query_emb, u) for u in unit_embs)


def type_specific_score(
    query_emb: list[float],
    unit_type: str,
    text_emb: list[float] | None,
    visual_emb: list[float] | None,
    ctx_emb: list[float] | None,
    *,
    gamma: float = 0.7,
) -> float:
    if unit_type == "text":
        if text_emb is None:
            return 0.0
        return cosine(query_emb, text_emb)
    vis = cosine(query_emb, visual_emb) if visual_emb is not None else 0.0
    ctx = cosine(query_emb, ctx_emb) if ctx_emb is not None else 0.0
    return gamma * vis + (1 - gamma) * ctx


def final_section_score(
    doc_score: float,
    sec_score: float,
    *,
    lam: float = 0.5,
) -> float:
    return lam * doc_score + (1.0 - lam) * sec_score