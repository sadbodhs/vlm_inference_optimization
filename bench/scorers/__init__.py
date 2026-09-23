"""Task metrics. Frozen early on purpose: changing a scorer mid-project invalidates
every Pareto point measured before the change (PLAN.md 8)."""
from .anls import anls, anls_score
from .multichoice import multichoice
from .relaxed import exact_match, relaxed_accuracy

SCORERS = {
    "anls": anls_score,          # DocVQA
    "relaxed": relaxed_accuracy,  # ChartQA
    "em": exact_match,            # VQA-style
    "mc": multichoice,            # TempCompass / video multiple choice
}

__all__ = ["anls", "anls_score", "relaxed_accuracy", "exact_match",
           "multichoice", "SCORERS"]
