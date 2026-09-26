#!/usr/bin/env python3
"""R2 (RECIPE.md): score the HA4M arms and check R2.1-R2.9.

Accuracy: the answer's first number equals the step (0-12). Chance: the arm's
answers shuffled across its samples (20 shuffles; 4 per bootstrap draw). Lift:
accuracy minus chance. Arms are compared by a paired bootstrap over workers.

    python3 experiments/r2_report.py --samples data/ha4m/r2_samples.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics as st
from pathlib import Path

PART = set(range(1, 9)) | {10, 11}
WHOLE = {9, 12}


def parse(text: str) -> int | None:
    m = re.search(r"\b(1[0-2]|[0-9])\b", text or "")
    return int(m.group(1)) if m else None


def load(d: Path):
    if not (d / "outputs.jsonl").exists():
        return None
    ans, tok = {}, []
    for line in (d / "outputs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["ok"]:
            ans[r["sample_id"]] = parse(r["text"])
    for line in (d / "latency.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("prompt_tokens") is not None:
            tok.append(r["prompt_tokens"])
    return ans, (st.mean(tok) if tok else None)


def lift(samples, ans, k=20, seed=0):
    ss = [s for s in samples if s["id"] in ans]
    if not ss:
        return None
    acc = sum(ans[s["id"]] == s["step"] for s in ss) / len(ss)
    pool = [ans[s["id"]] for s in ss]
    rng = random.Random(seed)
    ch = []
    for _ in range(k):
        rng.shuffle(pool)
        ch.append(sum(p == s["step"] for p, s in zip(pool, ss)) / len(ss))
    return acc - st.mean(ch), acc, st.mean(ch)


def paired(samples, a, b, boot=400, seed=7):
    subs = sorted({s["subject"] for s in samples})
    by = {x: [s for s in samples if s["subject"] == x] for x in subs}
    pt = lift(samples, a)[0] - lift(samples, b)[0]
    rng, d = random.Random(seed), []
    for i in range(boot):
        draw = [s for x in (rng.choice(subs) for _ in subs) for s in by[x]]
        la, lb = lift(draw, a, 4, i), lift(draw, b, 4, i)
        if la and lb:
            d.append(la[0] - lb[0])
    d.sort()
    return pt, d[int(.025 * len(d))], d[int(.975 * len(d)) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/r2")
    ap.add_argument("--samples", default="data/ha4m/r2_samples.json")
    ap.add_argument("--out", default="results/r2/report.json")
    args = ap.parse_args()
    root = Path(args.root)
    samples = [dict(s, id=f"{s['rec']}|{s['step']}|{s['frames'][0]}")
               for s in json.loads(Path(args.samples).read_text()) if not s["held_out"]]
    res = {"n": len(samples), "models": {}}
    subsets = {"part steps": lambda s: s["step"] in PART, "steps 9 and 12": lambda s: s["step"] in WHOLE,
               "lab": lambda s: s["setup"] == 1, "white room": lambda s: s["setup"] == 2}
    for model in ("E_q3vl_8b", "E_q3vl_4b"):
        runs = {n: load(root / f"{n}-{model}") for n in ("F", "B", "HK", "HY", "HKC", "F-n", "HK-n")}
        runs = {n: r for n, r in runs.items() if r}
        if "F" not in runs:
            continue
        F = runs["F"][0]
        out = {}
        print(f"\n=== {model} ===")
        print(f"{'arm':5s} {'acc':>6s} {'chance':>7s} {'lift':>6s} {'tokens':>7s} {'vs F [95%]':>22s}   "
              + "  ".join(f"{k:>14s}" for k in subsets))
        for n, (ans, tok) in runs.items():
            L = lift(samples, ans)
            pv = paired(samples, ans, F) if n != "F" else None
            row = {"accuracy": L[1], "chance": L[2], "lift": L[0], "prompt_tokens": tok, "vs_F": pv,
                   "answered_0": sum(v == 0 for v in ans.values()) / len(ans), "by": {}}
            for k, fn in subsets.items():
                ss = [s for s in samples if fn(s)]
                la, lf = lift(ss, ans), lift(ss, F)
                row["by"][k] = {"lift": la[0], "lift_F": lf[0], "gain": la[0] - lf[0], "n": len(ss)}
            out[n] = row
            vs = "" if not pv else f"{100*pv[0]:+5.1f} [{100*pv[1]:+5.1f},{100*pv[2]:+5.1f}]"
            print(f"{n:5s} {100*L[1]:6.1f} {100*L[2]:7.1f} {100*L[0]:+6.1f} {tok or 0:7.0f} {vs:>22s}   "
                  + "  ".join(f"{100*row['by'][k]['lift']:+7.1f} ({100*row['by'][k]['gain']:+5.1f})" for k in subsets))
        # per-step accuracy for F and HK
        for n in ("F", "HK"):
            if n in runs:
                a = runs[n][0]
                out[n]["per_step"] = {st_: sum(a.get(s["id"]) == st_ for s in samples if s["step"] == st_) /
                                           max(1, sum(s["step"] == st_ for s in samples)) for st_ in range(13)}
        res["models"][model] = out

    m, m4 = res["models"].get("E_q3vl_8b", {}), res["models"].get("E_q3vl_4b", {})
    if {"F", "B", "HK", "HY", "HKC", "F-n", "HK-n"} <= set(m):
        L = lambda a: m[a]["lift"]
        g = lambda a, k: m[a]["by"][k]["gain"]
        v = {"R2.1": {"holds": L("HK") - L("F") >= 0.05, "HK_minus_F": L("HK") - L("F"), "ci": m["HK"]["vs_F"]},
             "R2.2": {"holds": abs(L("HY") - L("HK")) <= 0.05, "HY_minus_HK": L("HY") - L("HK")},
             "R2.3": {"holds": g("HK", "part steps") > g("HK", "steps 9 and 12"),
                      "gain_part": g("HK", "part steps"), "gain_9_12": g("HK", "steps 9 and 12")},
             "R2.4": {"holds": g("HK", "white room") > g("HK", "lab"),
                      "gain_white": g("HK", "white room"), "gain_lab": g("HK", "lab")},
             "R2.5": {"holds": L("B") - L("F") < L("HK") - L("F"), "B_minus_F": L("B") - L("F")},
             "R2.6": {"holds": L("HKC") >= L("HK"), "HKC_minus_HK": L("HKC") - L("HK")},
             "R2.7": {"holds": L("F") - L("F-n") >= 0.10 and L("HK") - L("HK-n") >= 0.10,
                      "sheet_F": L("F") - L("F-n"), "sheet_HK": L("HK") - L("HK-n")},
             "R2.8": {"holds": m["HK"]["prompt_tokens"] / m["F"]["prompt_tokens"] <= 0.70,
                      "ratio": m["HK"]["prompt_tokens"] / m["F"]["prompt_tokens"]}}
        if {"F", "HK"} <= set(m4):
            v["R2.9"] = {"holds": m4["HK"]["lift"] - m4["F"]["lift"] >= L("HK") - L("F"),
                         "gain_4b": m4["HK"]["lift"] - m4["F"]["lift"], "gain_8b": L("HK") - L("F")}
        res["predictions"] = v
        print("\n=== predictions ===")
        for k, x in v.items():
            print(f"  {k}: {'HELD' if x['holds'] else 'failed'}  " + json.dumps(
                {a: (round(b, 3) if isinstance(b, float) else b) for a, b in x.items() if a != "holds"},
                default=lambda o: [round(y, 3) for y in o]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
