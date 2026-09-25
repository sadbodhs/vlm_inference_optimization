#!/usr/bin/env python3
"""R1b (RECIPE.md): score the five arms as the cascade runs, and check R1b.1-R1b.5.

Per activity instance (E7): recognised if its group letter is in the answer of any
fired window it overlaps -- for GS/GSC the union of that window's group answers.
Non-fired windows answer nothing in every arm. Chance: window answers shuffled
across fired windows (20 shuffles; 4 per bootstrap draw). False alarms per hour:
letters absent from the window's labels, over all 2 h of video. Tokens per fired
window: the sum over that window's requests.

    python3 experiments/r1b_report.py
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
from pathlib import Path

from e7_report import parse_answer

HOURS = 3600 * 60 / 30 / 3600          # 3,600 two-second windows
ratio = lambda a, b: a / b if b else float("inf") if a else 1.0


def load(d: Path):
    if not (d / "outputs.jsonl").exists():
        return None
    ans, tok = collections.defaultdict(set), collections.Counter()
    per_group = {}
    for line in (d / "outputs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not r["ok"]:
            continue
        wid = r["sample_id"].rsplit("|", 1)[0]
        a = parse_answer(r["text"])
        ans[wid] |= a
        per_group[r["sample_id"]] = a
    for line in (d / "latency.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("prompt_tokens") is not None:
            tok[r["sample_id"].rsplit("|", 1)[0]] += r["prompt_tokens"]
    return dict(ans), tok, per_group


CLIPS: list[str] = []      # all 24 clips, from the selection -- NOT from the activities:
                           # verified-empty cameras have none, and their false alarms count


def score(acts, wins, ans, clips=None, k=20, seed=0):
    """(lift, recognised, chance, false alarms per hour) over `clips` (a draw may repeat)."""
    clips = clips or CLIPS
    cnt = collections.Counter(clips)
    A = [a for a in acts if a["clip"] in cnt for _ in range(cnt[a["clip"]])]
    W = [w for w in wins if w["clip"] in cnt for _ in range(cnt[w["clip"]])]
    key = lambda clip, w: f"{clip}|{w}"

    def rec(m):
        if not A:                  # a draw of only verified-empty clips has no activities
            return 0.0
        return sum(any(a["group"] in m.get(key(a["clip"], w), set()) for w in a["fired"])
                   for a in A) / len(A)
    r = rec(ans)
    ids = [w["id"] for w in W]
    pool = [ans.get(i, set()) for i in ids]
    rng = random.Random(seed)
    ch = []
    for _ in range(k):
        rng.shuffle(pool)
        ch.append(rec(dict(zip(ids, pool))))
    fa = sum(len(ans.get(w["id"], set()) - set(w["truth"])) for w in W) / (HOURS * len(clips) / 24)
    return r - st.mean(ch), r, st.mean(ch), fa


def paired(acts, wins, a, b, boot=400, seed=7):
    clips = CLIPS
    rng, d = random.Random(seed), []
    for i in range(boot):
        draw = [rng.choice(clips) for _ in clips]
        d.append(score(acts, wins, a, draw, 4, i)[0] - score(acts, wins, b, draw, 4, i)[0])
    d.sort()
    return score(acts, wins, a)[0] - score(acts, wins, b)[0], d[int(.025 * len(d))], d[int(.975 * len(d)) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/r1b")
    ap.add_argument("--out", default="results/r1b/report.json")
    ap.add_argument("--selection", default="results/e7/selection.json")
    args = ap.parse_args()
    CLIPS[:] = sorted(c["clip"] for c in json.load(open(args.selection))["clips"])
    root = Path(args.root)
    W = json.loads((root / "windows.json").read_text())
    wins, acts = W["windows"], W["activities"]
    res = {"meta": W["meta"], "models": {}}
    for model in ("E_q3vl_4b", "E_q3vl_8b"):
        runs = {n: load(root / f"{n}-{model}") for n in ("F", "M", "GS", "GSC", "GO")}
        runs = {n: r for n, r in runs.items() if r}
        if "F" not in runs:
            continue
        out = {}
        print(f"\n=== {model} ===")
        print(f"{'arm':4s} {'lift':>6s} {'recog':>6s} {'chance':>7s} {'FA/h':>7s} {'tok/win':>8s} "
              f"{'vs F [95%]':>22s}")
        for n, (ans, tok, per_group) in runs.items():
            L = score(acts, wins, ans)
            pv = paired(acts, wins, ans, runs["F"][0]) if n != "F" else None
            t = st.mean(tok[w["id"]] for w in wins)
            out[n] = {"lift": L[0], "recognised": L[1], "chance": L[2], "fa_per_hour": L[3],
                      "tokens_per_window": t, "vs_F": pv}
            # attributed (GS/GSC): the letter came from a crop holding the actor
            if n in ("GS", "GSC"):
                hit = tot = 0
                for w in wins:
                    for c in w["covered"]:
                        if not c["crops"]:
                            continue
                        tot += 1
                        hit += any(c["group"] in per_group.get(f"{w['id']}|g{i}", set()) for i in c["crops"])
                out[n]["attributed_hit_rate"] = hit / tot if tot else None
            vs = "" if not pv else f"{100*pv[0]:+5.1f} [{100*pv[1]:+5.1f},{100*pv[2]:+5.1f}]"
            print(f"{n:4s} {100*L[0]:+6.1f} {100*L[1]:6.1f} {100*L[2]:7.1f} {L[3]:7.0f} {t:8.0f} {vs:>22s}")
        res["models"][model] = out

    m, m8 = res["models"].get("E_q3vl_4b", {}), res["models"].get("E_q3vl_8b", {})
    if {"F", "M", "GS", "GSC", "GO"} <= set(m):
        v = {}
        v["R1b.1"] = {"holds": m["GSC"]["lift"] - m["F"]["lift"] >= 0.05,
                      "GSC_minus_F": m["GSC"]["lift"] - m["F"]["lift"], "ci": m["GSC"]["vs_F"]}
        v["R1b.2"] = {"holds": m["GS"]["fa_per_hour"] >= 2 * m["F"]["fa_per_hour"]
                      and m["GSC"]["fa_per_hour"] <= 1.5 * m["F"]["fa_per_hour"],
                      "GS_over_F": ratio(m["GS"]["fa_per_hour"], m["F"]["fa_per_hour"]),
                      "GSC_over_F": ratio(m["GSC"]["fa_per_hour"], m["F"]["fa_per_hour"])}
        v["R1b.3"] = {"holds": abs(m["GO"]["lift"] - m["GSC"]["lift"]) <= 0.05
                      and m["GO"]["tokens_per_window"] <= 0.6 * m["GSC"]["tokens_per_window"]
                      and m["GO"]["tokens_per_window"] <= m["F"]["tokens_per_window"],
                      "GO_minus_GSC": m["GO"]["lift"] - m["GSC"]["lift"],
                      "GO_tokens_over_GSC": m["GO"]["tokens_per_window"] / m["GSC"]["tokens_per_window"],
                      "GO_tokens_over_F": m["GO"]["tokens_per_window"] / m["F"]["tokens_per_window"]}
        v["R1b.4"] = {"holds": m["M"]["lift"] >= m["F"]["lift"]
                      and abs(ratio(m["M"]["fa_per_hour"], m["F"]["fa_per_hour"]) - 1) <= 0.25
                      and m["M"]["tokens_per_window"] / m["F"]["tokens_per_window"] - 1 < 0.05,
                      "M_minus_F": m["M"]["lift"] - m["F"]["lift"],
                      "fa_ratio": ratio(m["M"]["fa_per_hour"], m["F"]["fa_per_hour"]),
                      "token_increase": m["M"]["tokens_per_window"] / m["F"]["tokens_per_window"] - 1}
        if {"F", "GO"} <= set(m8):
            g4, g8 = m["GO"]["lift"] - m["F"]["lift"], m8["GO"]["lift"] - m8["F"]["lift"]
            v["R1b.5"] = {"holds": g8 < g4, "gain_4b": g4, "gain_8b": g8}
        res["predictions"] = v
        print("\n=== predictions ===")
        for k, x in v.items():
            print(f"  {k}: {'HELD' if x['holds'] else 'failed'}  " + json.dumps(
                {a: (round(b, 3) if isinstance(b, float) else b) for a, b in x.items() if a != "holds"},
                default=lambda o: [round(y, 3) for y in o]))
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
