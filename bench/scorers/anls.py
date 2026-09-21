"""ANLS -- the official DocVQA metric.

Average Normalized Levenshtein Similarity against the closest ground-truth answer,
thresholded at 0.5: near-misses below the threshold score zero rather than partial
credit, so OCR failure does not get graded on a curve.
"""
from __future__ import annotations


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, gold: str) -> float:
    p, g = pred.strip().lower(), gold.strip().lower()
    if not p and not g:
        return 1.0
    denom = max(len(p), len(g))
    if denom == 0:
        return 0.0
    return 1.0 - _levenshtein(p, g) / denom


def anls_score(pred: str, golds: list[str], threshold: float = 0.5) -> float:
    if not golds:
        return 0.0
    best = max(anls(pred, g) for g in golds)
    return best if best >= threshold else 0.0
