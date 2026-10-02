#!/usr/bin/env python3
"""R1c (RECIPE.md): a precision pass on R1b's gate-fired windows.

The first pass is R1b's (results/r1b/<F|GO>-<model>/outputs.jsonl), reused unchanged.

  Fc, GOc   one pass: F's or GO's request with a stricter prompt (replaces the first pass)
  FV, GOV   for each window whose first answer named letters: one more request with the
            same images, yes/no for each named letter only
  F4V8      the 4B's F answers verified by THIS server's model (run it on the 8B)

    docker/run_harness.sh python3 experiments/r1c_run.py --arm arms/E_q3vl_4b.yaml --arms Fc,GOc,FV,GOV
"""
from __future__ import annotations

import asyncio
import collections
import json
from pathlib import Path

from _common import base_parser, resolve_arm
from e7_report import parse_answer
from e7_vlm import GROUPS, QUESTION
from r1b_run import encode, q_go, samples_for

STRICT = ("Most moments from this camera show none of these. Name a letter only if you "
          "can clearly see it happening; if unsure, leave it out.\n")
ANSWER_LINE = "Answer with every letter that applies"
ALL = ["Fc", "GOc", "FV", "GOV", "F4V8"]


def strict(q: str) -> str:
    i = q.index(ANSWER_LINE)
    return q[:i] + STRICT + q[i:]


def verify_q(sample_id: str, k_groups: int, letters: list[str]) -> str:
    """Yes/no for each named letter, framed like the first-pass request it checks."""
    if sample_id.endswith("|all"):
        head = q_go(k_groups).split("\nWhich of the following")[0]
        where = "in any of the close-ups"
    else:
        head = QUESTION.split("\nWhich of the following")[0]
        where = "in these frames"
    items = "\n".join(f"{g}. {GROUPS[g][0][0].upper()}{GROUPS[g][0][1:]}" for g in letters)
    form = ", ".join(f"{g}: yes" if i == 0 else f"{g}: no" for i, g in enumerate(letters))
    return (f"{head}\nLook carefully. For each statement below, answer yes only if you can "
            f"clearly see it happening {where}; otherwise answer no.\n{items}\n"
            f"Answer in this form: {form}")


def first_pass(root: Path, run: str) -> dict[str, list[str]]:
    ans = collections.defaultdict(set)
    for line in (root / run / "outputs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["ok"]:
            ans[r["sample_id"].rsplit("|", 1)[0]] |= parse_answer(r["text"])
    return {w: sorted(a) for w, a in ans.items() if a}


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--windows", default="results/r1b/windows.json")
    p.add_argument("--pass1-root", default="results/r1b")
    p.add_argument("--frame-dir", default="data/meva/frames")
    p.add_argument("--arms", default="Fc,GOc,FV,GOV")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="windows, for a smoke run")
    p.add_argument("--r1c-out", default="results/r1c")
    args = p.parse_args()

    from bench.data import Sample
    from bench.harness import run_arm
    arm = resolve_arm(args)
    arm.max_pixels = None
    wins = json.loads(Path(args.windows).read_text())["windows"]
    if args.limit:
        wins = wins[:args.limit]
    loop = asyncio.get_running_loop()
    for name in args.arms.split(","):
        base = {"Fc": "F", "GOc": "GO", "FV": "F", "GOV": "GO", "F4V8": "F"}[name]
        verify = name.endswith("V") or name == "F4V8"
        if verify:
            src = f"{base}-E_q3vl_4b" if name == "F4V8" else f"{base}-{arm.id}"
            named = first_pass(Path(args.pass1_root), src)
        arm.max_tokens = 40 if verify else 16

        def build(name=name, base=base):
            out = []
            for w in wins:
                (sid, q, imgs), = samples_for(base, w, args.frame_dir)   # F and GO: one request
                if verify:
                    letters = named.get(w["id"])
                    if not letters:
                        continue
                    q = verify_q(sid, len(w["crops"]), letters)
                else:
                    q = strict(q)
                frames, px = encode(imgs)
                out.append(Sample(id=sid, question=q, answers=w["truth"] or ["N"],
                                  frames_b64=frames, image_mime="jpeg", image_px=px))
            return out
        samples = await loop.run_in_executor(None, build)
        summary = await run_arm(
            arm, samples, mode="closed", concurrency=args.concurrency, scorer=None,
            results_root=args.r1c_out, run_id=f"{name}-{arm.id}",
            unmeasured=["latency under load (closed loop, not a capacity measurement)"])
        print(f"  {name:4s} requests {len(samples):5d}  prompt tokens mean "
              f"{summary['prompt_tokens']['mean']:.0f}  ok {summary['n_ok']}/{len(samples)}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
