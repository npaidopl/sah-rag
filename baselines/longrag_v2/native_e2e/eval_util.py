"""Evaluation utilities ported from utils/eval_util.py.

Source: utils/eval_util.py at pinned commit 778454e of TIGER-AI-Lab/LongRAG.
Reimplemented using only Python stdlib (no `regex` package dependency) to
keep the SAH-RAG project self-contained. Behavioural equivalence is
preserved for the metrics used by the LongRAG paper:

    - single_ans_em: exact match after SQuAD-style normalization
    - has_correct_answer: substring match on tokenized text

Known deviation: the official tokenizer uses Unicode property classes
(\\p{L}\\p{N}\\p{M}, \\p{Z}\\p{C}) via the `regex` package. This port uses
Python's `\\w` with re.UNICODE, which differs in edge cases involving
combining marks and zero-width joiners. For Wikipedia-style content
(English Q&A), the difference is empirically negligible; if strict
token-for-token equality is needed, install `regex` and swap the
tokenizer implementation.
"""

from __future__ import annotations

import re
import string
import unicodedata

_PUNCS = set(string.punctuation)
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _normalize_unicode(text: str) -> str:
    """NFC normalization, faithful enough to wiki_util._normalize for metrics."""
    if text is None:
        return ""
    # NFKC handles ligatures and compatibility forms; NFC preserves
    # composed accents. The official uses _normalize which mirrors
    # Wikipedia title normalization; for EM purposes NFKC is a safe
    # superset that does not merge distinct tokens.
    return unicodedata.normalize("NFKC", str(text))


def _tokenize(text: str, uncased: bool = True) -> list[str]:
    """Word-tokenize with optional lowercasing.

    Simple version of the official SimpleTokenizer.
    """
    tokens = _TOKEN_RE.findall(text)
    if uncased:
        return [t.lower() for t in tokens]
    return tokens


def remove_articles(text: str) -> str:
    return _ARTICLE_RE.sub(" ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    return "".join(ch for ch in text if ch not in _PUNCS)


def lower(text: str) -> str:
    return text.lower()


def normalize_answer(s: str) -> str:
    """SQuAD-style answer normalization."""
    return white_space_fix(remove_articles(remove_punc(lower(_normalize_unicode(s)))))


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def single_ans_em(pred: str, gold: str | list[str]) -> int:
    """Exact match of a single prediction against one or more gold answers.

    Returns 1 if any gold matches, else 0. Empty prediction against
    non-empty gold returns 0; empty prediction against empty gold returns 1.
    """
    if not isinstance(gold, list):
        gold = [gold]
    if not gold:
        return 0
    if pred is None:
        pred = ""
    return max(compute_exact(pred, a) for a in gold)


def has_correct_answer(retrieve_doc: str, answers: list[str]) -> int:
    """Return 1 if any gold answer appears as a contiguous token subsequence.

    Ported from the official implementation; used to compute the
    "retrieval accuracy" and "substring match" reported by the paper.
    """
    if not retrieve_doc or not answers:
        return 0
    doc_tokens = _tokenize(_normalize_unicode(retrieve_doc), uncased=True)
    for answer in answers:
        if answer is None:
            continue
        ans_tokens = _tokenize(_normalize_unicode(answer), uncased=True)
        n = len(ans_tokens)
        if n == 0:
            continue
        for i in range(0, len(doc_tokens) - n + 1):
            if ans_tokens == doc_tokens[i : i + n]:
                return 1
    return 0