#!/usr/bin/env python3
"""E7d: recognition across VLMs, paired against Qwen2.5-VL-7B (PLAN.md 13).

Every window, every clip, the same frames -- only the model differs -- so the
comparison is paired: each bootstrap draw resamples CLIPS once and scores both
models on that draw. The difference in lift over chance is what is reported; a
model is "not significantly below" the 7B if the 95% interval reaches 0.

Also reported per model: lift on phones (E) and conversations (F) (prediction 15),
false alarms per hour, and format compliance -- the share of answers that begin
with an option letter or N, as the prompt asks. A model that answers in prose is
scored as asserting nothing, which is what a deployment parsing its reply would get.

    docker/run_harness.sh python3 experiments/e7d_compare.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from e7_report import (LETTERS, chance_recognised, load_answers, parse_answer,  # noqa: F401
                       recognised)
from e7_vlm import TYPE_TO_GROUP, WINDOW

BASE = "V_vllm_video"
ARMS = [BASE, "E_q25_3b", "E_q3vl_8b", "E_q3vl_4b", "E_q3vl_2b",
        "E_q35_9b", "E_q35_4b", "E_q35_2b"]
LABEL = {BASE: "Qwen2.5-VL-7B", "E_q25_3b": "Qwen2.5-VL-3B",
         "E_q3vl_8b": "Qwen3-VL-8B", "E_q3vl_4b": "Qwen3-VL-4B", "E_q3vl_2b": "Qwen3-VL-2B",
         "E_q35_9b": "Qwen3.5-9B", "E_q35_4b": "Qwen3.5-4B", "E_q35_2b": "Qwen3.5-2B"}


def compliance(run_dir: Path) -> float | None:
    ok = n = 0
    f = run_dir / "outputs.jsonl"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("ok", True):
            continue
        toks = re.findall(r"[A-Za-z]+", (r.get("text") or "").strip())
        n += 1
        ok += bool(toks) and (toks[0].upper() in ("N", "NONE") or
                              (len(toks[0]) == 1 and toks[0].upper() in LETTERS
                               and bool(parse_answer(r["text"]))))
    return ok / n if n else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--out", default="results/e7d/compare.json")
    args = ap.parse_args()

    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    n_win = {c["clip"]: -(-idx[c["clip"]]["n_frames"] // WINDOW) for c in sel}
    dense = {c["clip"]: [True] * n_win[c["clip"]] for c in sel}
    hours = sum(n_win.values()) * WINDOW / 30 / 3600

    answers = {}
    for a in ARMS:
        full, _ = load_answers(Path(args.results) / f"e7-vlm-full-{a}")
        roi, _ = load_answers(Path(args.results) / f"e7-vlm-roi-{a}")
        if not full:
            print(f"  skip {a}: no full-frame run")
            continue
        answers[(a, "full")] = full
        # ROI windows with nothing to crop fall back to that model's full frame (as E7)
        answers[(a, "roi")] = {**full, **roi} if roi else None

    def answer_fn(ans):
        return lambda cl, w: ans.get(f"{cl.split('#')[0]}|{w}", set())

    def lift(clips, ans, k, seed):
        idx2 = {c["clip"]: idx[c["clip"].split("#")[0]] for c in clips}
        fired2 = {c["clip"]: dense[c["clip"].split("#")[0]] for c in clips}
        f = answer_fn(ans)
        h, n, per = recognised(clips, idx2, fired2, f)
        ch, ch_g = chance_recognised(clips, idx2, fired2, f, k=k, seed=seed)
        return h / n - ch, per, ch_g

    rows = []
    for which in ("full", "roi"):
        base = answers.get((BASE, which))
        for a in ARMS:
            ans = answers.get((a, which))
            if ans is None or base is None:
                continue
            pt, per, ch_g = lift(sel, ans, 20, 0)
            fa = 0
            for c in sel:
                acts = idx[c["clip"]]["activities"]
                for w in range(n_win[c["clip"]]):
                    lo, hi = w * WINDOW, (w + 1) * WINDOW - 1
                    truth = {TYPE_TO_GROUP[x["type"]] for x in acts
                             if x["start"] <= hi and x["end"] >= lo and x["type"] in TYPE_TO_GROUP}
                    fa += len(ans.get(f"{c['clip']}|{w}", set()) - truth)
            diff_ci = None
            if a != BASE:
                rng, d = random.Random(7), []
                for i in range(args.boot):
                    draw = [rng.choice(sel) for _ in sel]
                    clips = [dict(c, clip=f"{c['clip']}#{j}") for j, c in enumerate(draw)]
                    la, _, _ = lift(clips, ans, 4, 100 + i)
                    lb, _, _ = lift(clips, base, 4, 100 + i)
                    d.append(la - lb)
                d.sort()
                diff_ci = [d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]]
            rows.append({
                "arm_id": a, "model": LABEL[a], "view": which, "lift": pt,
                "lift_vs_7b": None if a == BASE else pt - rows_base_lift(rows, which),
                "lift_vs_7b_ci95": diff_ci,
                "false_alarms_per_hour": fa / hours,
                "compliance": compliance(Path(args.results) / f"e7-vlm-{which}-{a}"),
                "per_group": {g: {"n": v[1], "recognised": v[0] / v[1],
                                  "chance": ch_g.get(g, 0.0), "lift": v[0] / v[1] - ch_g.get(g, 0.0)}
                              for g, v in sorted(per.items()) if v[1]},
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"base": BASE, "boot": args.boot, "hours": hours,
                        "note": "dense (every window); paired clip bootstrap of lift difference"},
               "rows": rows}, open(args.out, "w"), indent=1)

    pct = lambda v: "    -" if v is None else f"{100 * v:+5.1f}"
    for which in ("full", "roi"):
        print(f"\n=== {which} ===")
        print(f"{'model':15s} {'lift':>6s} {'vs 7B':>6s} {'95% CI':>15s} {'FA/h':>7s} "
              f"{'format':>7s} {'E phone':>8s} {'F talk':>7s}")
        for r in rows:
            if r["view"] != which:
                continue
            ci = r["lift_vs_7b_ci95"]
            ci_s = "" if ci is None else f"[{100*ci[0]:+.1f},{100*ci[1]:+.1f}]"
            pg = r["per_group"]
            print(f"{r['model']:15s} {pct(r['lift'])} {pct(r['lift_vs_7b'])} {ci_s:>15s} "
                  f"{r['false_alarms_per_hour']:7.0f} {100*(r['compliance'] or 0):6.1f}% "
                  f"{pct(pg.get('E', {}).get('lift')):>8s} {pct(pg.get('F', {}).get('lift')):>7s}")
    print(f"\nwrote {args.out}")


def rows_base_lift(rows, which):
    return next(r["lift"] for r in rows if r["arm_id"] == BASE and r["view"] == which)


if __name__ == "__main__":
    main()
