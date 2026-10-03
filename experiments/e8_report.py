#!/usr/bin/env python3
"""E8: recognition and cameras for images vs video vs EVS-pruned video (PLAN.md 14).

Recognition is scored exactly as E7d (e7d_compare.py): every window, lift =
recognised - shuffled-answer chance. Each arm is paired against the SAME model's
img2 -- each bootstrap draw resamples clips once and scores both on that draw --
so a difference counts only if its 95% interval excludes 0.

Cameras come from the live runs (e7c.py live --input ...): the highest supported
count in results/e8/<arm id>/<input>/track-motion-full-deepstream.json.

    docker/run_harness.sh python3 experiments/e8_report.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from e7_report import chance_recognised, load_answers, recognised
from e7_vlm import TYPE_TO_GROUP, WINDOW
from e7d_compare import compliance
from e8_vlm import run_name

MODELS = {"V_vllm_video": "Qwen2.5-VL-7B", "E_q3vl_4b": "Qwen3-VL-4B", "E_q3vl_8b": "Qwen3-VL-8B"}
# (label, input, evs rate); the order is the table order
ARMS = [("img2", "img2", 0.0), ("vid2", "vid2", 0.0), ("vid8", "vid8", 0.0),
        ("vid8-e50", "vid8", 0.5), ("vid8-e75", "vid8", 0.75)]
STATE_CHANGE, STATIC = ("A", "C", "G"), ("E", "F")      # prediction 18


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--out", default="results/e8/report.json")
    args = ap.parse_args()

    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    n_win = {c["clip"]: -(-idx[c["clip"]]["n_frames"] // WINDOW) for c in sel}   # as E7d
    dense = {c["clip"]: [True] * n_win[c["clip"]] for c in sel}
    hours = sum(n_win.values()) * WINDOW / 30 / 3600

    def lift(clips, ans, k, seed):
        idx2 = {c["clip"]: idx[c["clip"].split("#")[0]] for c in clips}
        fired2 = {c["clip"]: dense[c["clip"].split("#")[0]] for c in clips}
        f = lambda cl, w: ans.get(f"{cl.split('#')[0]}|{w}", set())
        h, n, per = recognised(clips, idx2, fired2, f)
        ch, ch_g = chance_recognised(clips, idx2, fired2, f, k=k, seed=seed)
        return h / n - ch, per, ch_g

    def cameras(arm_id, label):
        f = Path(args.results) / "e8" / arm_id / label / "track-motion-full-deepstream.json"
        if not f.exists():
            return None
        ok = [r["cams"] for r in json.loads(f.read_text())["rows"] if r["supported"]]
        return max(ok, default=0)

    rows = []
    for arm_id, model in MODELS.items():
        base_dir = Path(args.results) / run_name("img2", 0.0, arm_id)
        base, _ = load_answers(base_dir)
        for label, inp, evs in ARMS:
            run_dir = Path(args.results) / run_name(inp, evs, arm_id)
            ans, tok = load_answers(run_dir)
            if not ans:
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
            if label != "img2" and base:
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
                "arm_id": arm_id, "model": model, "arm": label, "evs": evs,
                "n_answers": len(ans), "tokens_per_call": tok, "lift": pt,
                "lift_vs_img2_ci95": diff_ci,
                "false_alarms_per_hour": fa / hours,
                "compliance": compliance(run_dir),
                "cameras": cameras(arm_id, label),
                "per_group": {g: {"n": v[1], "recognised": v[0] / v[1],
                                  "chance": ch_g.get(g, 0.0), "lift": v[0] / v[1] - ch_g.get(g, 0.0)}
                              for g, v in sorted(per.items()) if v[1]},
            })
        mrows = [r for r in rows if r["arm_id"] == arm_id]
        b = next((r for r in mrows if r["arm"] == "img2"), None)
        for r in mrows:
            if b and r is not b:
                r["lift_vs_img2"] = r["lift"] - b["lift"]
                r["tokens_vs_img2"] = r["tokens_per_call"] / b["tokens_per_call"] - 1
                r["group_lift_vs_img2"] = {g: r["per_group"][g]["lift"] - b["per_group"][g]["lift"]
                                           for g in r["per_group"] if g in b["per_group"]}

    # Prediction 19: share of vid8's gain over vid2 that the pruned arm keeps
    for arm_id in MODELS:
        by = {r["arm"]: r for r in rows if r["arm_id"] == arm_id}
        if {"vid2", "vid8"} <= set(by):
            gain = by["vid8"]["lift"] - by["vid2"]["lift"]
            for lab in ("vid8-e50", "vid8-e75"):
                if lab in by:
                    by[lab]["kept_of_vid8_gain"] = ((by[lab]["lift"] - by["vid2"]["lift"]) / gain
                                                    if gain > 0 else None)
                    by[lab]["tokens_vs_vid2"] = by[lab]["tokens_per_call"] / by["vid2"]["tokens_per_call"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"boot": args.boot, "hours": hours, "windows": sum(n_win.values()),
                        "note": "dense (every window); paired clip bootstrap vs the same model's img2"},
               "rows": rows}, open(args.out, "w"), indent=1)

    pct = lambda v: "    -" if v is None else f"{100 * v:+5.1f}"
    print(f"{'model':14s} {'arm':9s} {'tokens':>7s} {'lift':>6s} {'vs img2':>8s} {'95% CI':>15s} "
          f"{'FA/h':>6s} {'format':>7s} {'A,C,G':>6s} {'E,F':>6s} {'cams':>5s}")
    for r in rows:
        ci = r["lift_vs_img2_ci95"]
        g = r.get("group_lift_vs_img2", {})
        sc = [g[k] for k in STATE_CHANGE if k in g]
        st = [g[k] for k in STATIC if k in g]
        print(f"{r['model']:14s} {r['arm']:9s} {r['tokens_per_call']:7.0f} {pct(r['lift'])} "
              f"{pct(r.get('lift_vs_img2')):>8s} "
              f"{'' if ci is None else f'[{100*ci[0]:+.1f},{100*ci[1]:+.1f}]':>15s} "
              f"{r['false_alarms_per_hour']:6.0f} {100 * (r['compliance'] or 0):6.1f}% "
              f"{pct(sum(sc) / len(sc) if sc else None):>6s} {pct(sum(st) / len(st) if st else None):>6s} "
              f"{'-' if r['cameras'] is None else r['cameras']:>5}")
        if "kept_of_vid8_gain" in r:
            k = r["kept_of_vid8_gain"]
            print(f"{'':24s} keeps {'-' if k is None else f'{100*k:.0f}%'} of vid8's gain over vid2, "
                  f"at {r['tokens_vs_vid2']:.2f}x vid2's tokens")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
