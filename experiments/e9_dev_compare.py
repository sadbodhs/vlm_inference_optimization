#!/usr/bin/env python3
"""E9 step C: pruned-before-the-encoder answers vs E8's, on one clip per bin.

A go/no-go check, not the experiment: 4 clips (selection[::6], one per duty-cycle
bin), 600 windows, so the lift intervals are wide. What it can show is a large
loss, and whether pruned answers stay close to the unpruned 8-frame ones.

Per arm, on the same windows: recognition above chance (E7's lift), false alarms
per hour, mean prompt tokens, and agreement with unpruned 8 frames (share of
windows with the identical answer set, and with the same "something / nothing").

    python3 experiments/e9_dev_compare.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e7_report import chance_recognised, load_answers, recognised  # noqa: E402
from e7_vlm import TYPE_TO_GROUP, WINDOW  # noqa: E402

ARMS = [  # label, run dir
    ("2 images", "results/e8-img2-V_vllm_video"),
    ("2 frames as video", "results/e8-vid2-V_vllm_video"),
    ("8 frames", "results/e8-vid8-V_vllm_video"),
    ("8 frames, vLLM EVS 0.5", "results/e8-vid8-e50-V_vllm_video"),
    ("8 frames, pre-encoder unit 0.675", "results/e9/dev/acc-unit/e8-vid8-e68-V_vllm_video"),
    ("8 frames, pre-encoder window 0.675", "results/e9/dev/acc-window/e8-vid8-e68-V_vllm_video"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--out", default="results/e9/dev/compare.json")
    args = ap.parse_args()
    sel = json.load(open(args.selection))["clips"][::6][:4]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    clips = {c["clip"] for c in sel}
    n_win = {c["clip"]: -(-idx[c["clip"]]["n_frames"] // WINDOW) for c in sel}
    dense = {c["clip"]: [True] * n_win[c["clip"]] for c in sel}
    hours = sum(n_win.values()) * WINDOW / 30 / 3600

    loaded = {}
    for lab, d in ARMS:
        ans, _ = load_answers(Path(d))
        ans = {k: v for k, v in ans.items() if k.split("|")[0] in clips}
        if not ans:
            print(f"  skip {lab}: no answers in {d}")
            continue
        toks = []
        f = Path(d) / "latency.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r.get("sample_id", "").split("|")[0] in clips and r.get("prompt_tokens"):
                    toks.append(r["prompt_tokens"])
        loaded[lab] = (ans, sum(toks) / len(toks) if toks else None)
    ref = loaded.get("8 frames", (None,))[0]

    rows = []
    for lab, (ans, tok) in loaded.items():
        f = lambda cl, w, a=ans: a.get(f"{cl}|{w}", set())
        h, n, _ = recognised(sel, idx, dense, f)
        ch, _ = chance_recognised(sel, idx, dense, f, k=20, seed=0)
        fa = 0
        for c in sel:
            acts = idx[c["clip"]]["activities"]
            for w in range(n_win[c["clip"]]):
                lo, hi = w * WINDOW, (w + 1) * WINDOW - 1
                truth = {TYPE_TO_GROUP[x["type"]] for x in acts
                         if x["start"] <= hi and x["end"] >= lo and x["type"] in TYPE_TO_GROUP}
                fa += len(ans.get(f"{c['clip']}|{w}", set()) - truth)
        same = bool_same = None
        if ref is not None:
            keys = [k for k in ans if k in ref]
            same = sum(ans[k] == ref[k] for k in keys) / len(keys)
            bool_same = sum(bool(ans[k]) == bool(ref[k]) for k in keys) / len(keys)
        rows.append({"arm": lab, "windows": len(ans), "tokens": tok, "lift": h / n - ch,
                     "instances": n, "false_alarms_per_hour": fa / hours,
                     "same_answer_as_8_frames": same, "same_something_nothing": bool_same})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"clips": sorted(clips), "rows": rows}, open(args.out, "w"), indent=1)
    print(f"{len(clips)} clips, {sum(n_win.values())} windows, {rows[0]['instances']} activity instances")
    print(f"{'arm':36s} {'tokens':>7s} {'lift':>6s} {'FA/h':>6s} {'same as 8f':>11s} {'same y/n':>9s}")
    for r in rows:
        tok = "-" if r["tokens"] is None else f"{r['tokens']:.0f}"
        sa = "-" if r["same_answer_as_8_frames"] is None else f"{100*r['same_answer_as_8_frames']:.1f}%"
        sb = "-" if r["same_something_nothing"] is None else f"{100*r['same_something_nothing']:.1f}%"
        print(f"{r['arm']:36s} {tok:>7s} {100*r['lift']:+5.1f} {r['false_alarms_per_hour']:6.0f} "
              f"{sa:>11s} {sb:>9s}")


if __name__ == "__main__":
    main()
