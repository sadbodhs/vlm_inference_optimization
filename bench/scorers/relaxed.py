"""ChartQA relaxed accuracy and plain exact match."""
from __future__ import annotations

import re

_NUM = re.compile(r"-?\d+\.?\d*")


def _norm(s: str) -> str:
    s = s.strip().lower().rstrip(".")
    return re.sub(r"\s+", " ", s)


def _as_float(s: str) -> float | None:
    m = _NUM.search(s.replace(",", "").replace("%", ""))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def relaxed_accuracy(pred: str, golds: list[str], tol: float = 0.05) -> float:
    """Numeric answers count as correct within 5% relative error; text falls back
    to exact match. This is the ChartQA convention."""
    p_num = _as_float(pred)
    for g in golds:
        g_num = _as_float(g)
        if p_num is not None and g_num is not None:
            if g_num == 0:
                if abs(p_num) <= tol:
                    return 1.0
            elif abs(p_num - g_num) / abs(g_num) <= tol:
                return 1.0
        elif _norm(pred) == _norm(g):
            return 1.0
    return 0.0


def exact_match(pred: str, golds: list[str]) -> float:
    p = _norm(pred)
    return 1.0 if any(p == _norm(g) for g in golds) else 0.0
