"""Multiple-choice scoring for benchmarks like TempCompass.

Answers arrive as the full option, "A. dunking a basketball", while a model may
reply with the letter alone, the text alone, or a sentence containing either. All
three are the same answer and scoring only one of them measures prompt compliance
rather than understanding -- the mistake that made DocVQA read 0.03 ANLS while 92%
of predictions contained the gold span.

Chance level matters for reading the result: TempCompass multi-choice is mostly
3-option, so ~0.33 is the floor, not 0.
"""
from __future__ import annotations

import re

# A bare option letter, or a letter that OPENS the reply: "A", "A.", "(A)", "B) ...".
# Deliberately NOT a search for any standalone letter: in English "a" is an article,
# so "dribbling a basketball" would match gold option A and score a wrong answer
# correct. Matching must be anchored.
_LETTER_ONLY = re.compile(r"^\(?([A-Ea-e])\)?\s*[.):\-]?\s*$")
_LETTER_LEAD = re.compile(r"^\(?([A-Ea-e])\)?[.):]\s+")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().rstrip(".")).strip()


def _split_option(opt: str) -> tuple[str | None, str]:
    """'A. dunking a basketball' -> ('a', 'dunking a basketball')"""
    m = re.match(r"\s*([A-Ea-e])[.)]\s*(.*)", opt)
    if m:
        return m.group(1).lower(), _norm(m.group(2))
    return None, _norm(opt)


def multichoice(pred: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    g_letter, g_text = _split_option(golds[0])
    p = _norm(pred)

    # exact option text, or the text appearing inside a longer reply
    if g_text and (p == g_text or g_text in p):
        return 1.0

    # a bare letter, or a letter opening the reply -- anchored, never a mid-sentence
    # article
    if g_letter:
        raw = pred.strip()
        for rx in (_LETTER_ONLY, _LETTER_LEAD):
            m = rx.match(raw)
            if m and m.group(1).lower() == g_letter:
                return 1.0
    return 0.0
