#!/usr/bin/env python3
"""R1 (RECIPE.md): score every arm and check predictions R1.1-R1.7.

Recognised: the activity's group letter is in the answer (E7's parse_answer).
Lift: recognised minus chance, chance from shuffling the arm's own answers across
all its samples, positives and negatives together (20 shuffles; 4 per bootstrap
draw). False alarm: a negative answered with any letter. Differences between arms
are paired: each bootstrap draw resamples clips once and scores both arms on it.

    python3 experiments/r1_report.py            # after rsyncing results/r1 from the rig
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
from pathlib import Path

from e7_report import parse_answer

A2 = ["A2-m1.2", "A2-m1.5", "A2-m2", "A2-m3"]
A3 = ["A3-h112", "A3-h224", "A3-h448"]


def load_run(d: Path):
    ans, tok, fmt = {}, {}, {}
    if not (d / "outputs.jsonl").exists():
        return None
    for line in (d / "outputs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["ok"]:
            ans[r["sample_id"]] = parse_answer(r["text"])
            toks = re.findall(r"[A-Za-z]+", (r["text"] or "").strip())
            fmt[r["sample_id"]] = bool(toks) and (toks[0].upper() in ("N", "NONE") or
                                                  bool(parse_answer(r["text"])))
    for line in (d / "latency.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("prompt_tokens") is not None:
            tok[r["sample_id"]] = r["prompt_tokens"]
    return ans, tok, fmt


def lift(samples, ans, k=20, seed=0):
    """(lift, recognised, chance, false-alarm rate) over `samples` (dicts)."""
    pos = [s for s in samples if s["kind"] == "pos" and s["id"] in ans]
    neg = [s for s in samples if s["kind"] == "neg" and s["id"] in ans]
    if not pos:
        return None
    rec = sum(s["group"] in ans[s["id"]] for s in pos) / len(pos)
    pool = [ans[s["id"]] for s in pos + neg]
    rng = random.Random(seed)
    ch = []
    for _ in range(k):
        rng.shuffle(pool)
        ch.append(sum(s["group"] in a for s, a in zip(pos, pool)) / len(pos))
    fa = sum(bool(ans[s["id"]]) for s in neg) / len(neg) if neg else None
    return rec - st.mean(ch), rec, st.mean(ch), fa


def paired(samples, ans_a, ans_b, boot=400, seed=7):
    """Lift(a) - lift(b), with a 95% interval from resampling clips."""
    clips = sorted({s["clip"] for s in samples})
    by = {c: [s for s in samples if s["clip"] == c] for c in clips}
    pt = lift(samples, ans_a)[0] - lift(samples, ans_b)[0]
    rng, d = random.Random(seed), []
    for i in range(boot):
        draw = [s for c in (rng.choice(clips) for _ in clips) for s in by[c]]
        la, lb = lift(draw, ans_a, 4, i), lift(draw, ans_b, 4, i)
        if la and lb:
            d.append(la[0] - lb[0])
    d.sort()
    return pt, d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/r1")
    ap.add_argument("--out", default="results/r1/report.json")
    args = ap.parse_args()
    root = Path(args.root)
    S = json.loads((root / "samples.json").read_text())
    samples, meta = S["samples"], S["meta"]
    res = {"meta": meta, "models": {}}

    for model in ("E_q3vl_4b", "E_q3vl_8b"):
        runs = {}
        for d in sorted(root.glob(f"*-{model}")):
            r = load_run(d)
            if r:
                runs[d.name[: -len(model) - 1]] = r
        if "A0" not in runs:
            continue
        a0 = runs["A0"][0]
        out = {}
        print(f"\n=== {model} ===")
        print(f"{'arm':9s} {'n':>5s} {'lift':>6s} {'recog':>6s} {'chance':>7s} {'FA neg':>7s} "
              f"{'tokens':>7s} {'format':>7s} {'vs A0 [95%]':>22s}")
        for arm, (ans, tok, fmt) in runs.items():
            sub = [s for s in samples if s["id"] in ans]
            L = lift(sub, ans)
            # vs A0 on the samples both saw (A5 covers tracker-matched positives only)
            pa = paired(sub, ans, a0) if arm != "A0" else None
            t = st.mean(tok.values()) if tok else None
            row = {"n": len(sub), "lift": L[0], "recognised": L[1], "chance": L[2],
                   "false_alarm_rate": L[3], "prompt_tokens": t,
                   "format_ok": sum(fmt.values()) / len(fmt) if fmt else None,
                   "vs_A0": pa}
            # breakdowns, each against A0 on the same subset
            for key, pred in (("small (<100 px)", lambda s: s["kind"] == "neg" or (s["actor_h"] or 0) < 100),
                              ("large (>=100 px)", lambda s: s["kind"] == "neg" or (s["actor_h"] or 0) >= 100),
                              ("groups A+C", lambda s: s["kind"] == "neg" or s["group"] in ("A", "C")),
                              ("1 person", lambda s: s["kind"] == "neg" or s["cls"] == "1 person"),
                              ("2+ people", lambda s: s["kind"] == "neg" or s["cls"] == "2+ people"),
                              ("person + vehicle", lambda s: s["kind"] == "neg" or s["cls"] == "person + vehicle")):
                ss = [s for s in sub if pred(s)]
                la, lb = lift(ss, ans), lift(ss, a0)
                row.setdefault("by", {})[key] = {"lift": la[0] if la else None,
                                                 "lift_A0": lb[0] if lb else None,
                                                 "n_pos": sum(s["kind"] == "pos" for s in ss)}
            out[arm] = row
            vs = "" if not pa else f"{100*pa[0]:+5.1f} [{100*pa[1]:+5.1f},{100*pa[2]:+5.1f}]"
            print(f"{arm:9s} {len(sub):5d} {100*L[0]:+6.1f} {100*L[1]:6.1f} {100*L[2]:7.1f} "
                  f"{100*(L[3] or 0):6.1f}% {t or 0:7.0f} {100*(row['format_ok'] or 0):6.1f}% {vs:>22s}")
        # A5 against the ground-truth 2x crop on the SAME tracker-matched samples
        if "A5" in runs and "A2-m2" in runs:
            ids = set(runs["A5"][0])
            sub = [s for s in samples if s["id"] in ids]
            out["A5_vs_A2-m2_matched"] = paired(sub, runs["A5"][0], runs["A2-m2"][0])
        res["models"][model] = out

    # ---- predictions, on Qwen3-VL-4B (the arm set that has them all) ----
    m = res["models"].get("E_q3vl_4b", {})
    if m:
        L = lambda a: m[a]["lift"]
        v = {}
        best = max(A2, key=L)
        v["R1.1"] = {"best_margin": best,
                     "holds": best in ("A2-m1.5", "A2-m2") and L("A2-m1.2") < L(best)
                     and L("A2-m3") < L(best) and L(best) > L("A0"),
                     "lifts": {a: L(a) for a in ["A0"] + A2}}
        v["R1.2"] = {"holds": L("A3-h224") > L("A3-h112") and abs(L("A3-h448") - L("A3-h224")) < 0.03,
                     "lifts": {a: L(a) for a in A3}}
        g = m[best]["by"]
        gs = g["small (<100 px)"]["lift"] - g["small (<100 px)"]["lift_A0"]
        gl = g["large (>=100 px)"]["lift"] - g["large (>=100 px)"]["lift_A0"]
        v["R1.3"] = {"holds": gs > gl, "gain_small": gs, "gain_large": gl}
        ac4, ac2 = m["A4"]["by"]["groups A+C"]["lift"], m["A2-m2"]["by"]["groups A+C"]["lift"]
        v["R1.4"] = {"holds": ac4 >= ac2, "A4": ac4, "A2-m2": ac2}
        tok_ratio = m["A1"]["prompt_tokens"] / m["A0"]["prompt_tokens"] - 1
        v["R1.5"] = {"holds": L("A1") >= L("A0") and tok_ratio < 0.05, "token_increase": tok_ratio}
        r6 = m["A2-m2"]["prompt_tokens"] / m["A0"]["prompt_tokens"]
        v["R1.6"] = {"holds": r6 <= 0.5, "ratio": r6}
        frac = meta["tracked_positives"] / meta["positives"]
        d5 = m.get("A5_vs_A2-m2_matched")
        v["R1.7"] = {"holds": frac >= 0.6 and d5 is not None and abs(d5[0]) <= 0.05,
                     "tracked_fraction": frac, "A5_minus_gt": d5}
        res["predictions"] = v
        print("\n=== predictions (Qwen3-VL-4B) ===")
        for k, x in v.items():
            print(f"  {k}: {'HELD' if x['holds'] else 'failed'}  "
                  + json.dumps({a: (round(b, 3) if isinstance(b, float) else b)
                                for a, b in x.items() if a != 'holds'}, default=str))
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
