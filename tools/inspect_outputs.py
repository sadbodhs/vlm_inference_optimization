#!/usr/bin/env python3
"""Show generated text next to gold answers and the score each one earned.

An aggregate accuracy number cannot tell you the difference between "the model
cannot read the document" and "the model answered correctly in a full sentence
while the metric expects a bare span". Those demand opposite fixes, so look at
the rows before believing the mean.

    python3 tools/inspect_outputs.py results/e3-...-px1605632 --manifest data/docvqa/manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.scorers import SCORERS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scorer", default="anls")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.manifest).parent
    gold = {}
    for line in Path(args.manifest).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            gold[str(r["id"])] = r

    fn = SCORERS[args.scorer]
    rows = [json.loads(l) for l in
            (Path(args.run_dir) / "outputs.jsonl").read_text().splitlines() if l.strip()]

    scores = []
    for r in rows:
        g = gold.get(str(r["sample_id"]))
        if not g or not g.get("answers"):
            continue
        scores.append(fn(r["text"], g["answers"]))

    print(f"{len(scores)} scored, mean {args.scorer} = "
          f"{sum(scores)/len(scores):.3f}\n" if scores else "nothing scored\n")

    shown = 0
    for r in rows:
        g = gold.get(str(r["sample_id"]))
        if not g:
            continue
        s = fn(r["text"], g["answers"]) if g.get("answers") else None
        print(f"[{s if s is None else f'{s:.2f}'}] {g['question'][:70]}")
        print(f"    gold: {g.get('answers')}")
        print(f"    pred: {r['text'][:120]!r}")
        shown += 1
        if shown >= args.n:
            break

    # The diagnostic that matters: how often is the gold answer present inside a
    # longer prediction? High here with low ANLS means a formatting problem, not a
    # perception problem.
    contained = 0
    total = 0
    for r in rows:
        g = gold.get(str(r["sample_id"]))
        if not g or not g.get("answers"):
            continue
        total += 1
        pred = r["text"].strip().lower()
        if any(a.strip().lower() in pred for a in g["answers"]):
            contained += 1
    if total:
        print(f"\ngold answer appears inside the prediction: {contained}/{total} "
              f"({100*contained/total:.0f}%)")
        print(f"exact-span score ({args.scorer}):            "
              f"{sum(scores)/len(scores):.3f}" if scores else "")
        if contained / total > 0.5 and scores and sum(scores) / len(scores) < 0.3:
            print("\n  => The model is finding the answer and then wrapping it in a "
                  "sentence.\n     That is a PROMPT problem, not a vision problem: "
                  "DocVQA expects a bare\n     span. Fix the prompt, not the model.")


if __name__ == "__main__":
    main()
