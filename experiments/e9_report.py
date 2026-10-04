#!/usr/bin/env python3
"""E9: pruning before the vision encoder vs E8's inputs (PLAN.md 15).

Recognition is scored as E8 (every window, lift = recognised - shuffled chance).
Each arm is paired against three references by a bootstrap over clips: unpruned
8 frames, vLLM EVS 0.5 (pruned after the encoder, same token count as `pre-0.5`),
and two images. Also: false alarms per hour, prompt tokens, and agreement with
unpruned 8 frames (identical answer set; same something/nothing).

Cameras: highest supported count of the live sweep. E8's arms come from
results/e8/V_vllm_video/<label> (EVS 0.75 merged with its rerun), E9's from
results/e9/live/<label>. A count where requests failed is a server crash.

    python3 experiments/e9_report.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e7_report import chance_recognised, load_answers, recognised  # noqa: E402
from e7_vlm import TYPE_TO_GROUP, WINDOW  # noqa: E402

# label, recognition run, live sweep dir(s), source
ARMS = [
    ("2 images", "results/e8-img2-V_vllm_video", ["results/e8/V_vllm_video/img2"], "E8"),
    ("2 frames as video", "results/e8-vid2-V_vllm_video", ["results/e8/V_vllm_video/vid2"], "E8"),
    ("8 frames", "results/e8-vid8-V_vllm_video", ["results/e8/V_vllm_video/vid8"], "E8"),
    ("8 frames, vLLM EVS 0.5", "results/e8-vid8-e50-V_vllm_video", ["results/e9/live/evs-0.5"], "E8 / E9 live"),
    ("8 frames, vLLM EVS 0.75 (first pair only)", "results/e8-vid8-e75-V_vllm_video",
     ["results/e8/V_vllm_video/vid8-e75", "results/e8/V_vllm_video/vid8-e75-rerun"], "E8"),
    ("pre-encoder 0.675", "results/e9/e8-vid8-e68-V_vllm_video", ["results/e9/live/pre-0.675"], "E9"),
    ("pre-encoder 0.5", "results/e9/e8-vid8-e50-V_vllm_video", ["results/e9/live/pre-0.5"], "E9"),
    ("pre-encoder 0.675, tracker", "results/e9/e8-vid8t-e68-V_vllm_video", ["results/e9/live/pre-trk-0.675"], "E9"),
]
REFS = ("8 frames", "8 frames, vLLM EVS 0.5", "2 images")


def cameras(dirs):
    rows = {}
    for d in dirs:
        f = Path(d) / "track-motion-full-deepstream.json"
        if f.exists():
            rows.update({r["cams"]: r for r in json.loads(f.read_text())["rows"]})
    if not rows:
        return None, []
    best = max((n for n, r in rows.items() if r["supported"]), default=0)
    return best, sorted(n for n, r in rows.items() if r["ok"] < r["sent"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--out", default="results/e9/report.json")
    args = ap.parse_args()
    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    n_win = {c["clip"]: -(-idx[c["clip"]]["n_frames"] // WINDOW) for c in sel}   # as E7d/E8
    dense = {c["clip"]: [True] * n_win[c["clip"]] for c in sel}
    hours = sum(n_win.values()) * WINDOW / 30 / 3600

    def lift(clips, ans, k, seed):
        idx2 = {c["clip"]: idx[c["clip"].split("#")[0]] for c in clips}
        fired2 = {c["clip"]: dense[c["clip"].split("#")[0]] for c in clips}
        f = lambda cl, w: ans.get(f"{cl.split('#')[0]}|{w}", set())
        h, n, per = recognised(clips, idx2, fired2, f)
        ch, ch_g = chance_recognised(clips, idx2, fired2, f, k=k, seed=seed)
        return h / n - ch, per, ch_g

    answers = {}
    for lab, run, _, _ in ARMS:
        a, tok = load_answers(Path(run))
        if a:
            answers[lab] = (a, tok)
        else:
            print(f"  skip {lab}: no answers in {run}")

    rows = []
    for lab, run, live, src in ARMS:
        if lab not in answers:
            continue
        ans, tok = answers[lab]
        pt, per, ch_g = lift(sel, ans, 20, 0)
        fa = 0
        for c in sel:
            acts = idx[c["clip"]]["activities"]
            for w in range(n_win[c["clip"]]):
                lo, hi = w * WINDOW, (w + 1) * WINDOW - 1
                truth = {TYPE_TO_GROUP[x["type"]] for x in acts
                         if x["start"] <= hi and x["end"] >= lo and x["type"] in TYPE_TO_GROUP}
                fa += len(ans.get(f"{c['clip']}|{w}", set()) - truth)
        vs = {}
        for ref in REFS:
            if ref == lab or ref not in answers:
                continue
            base = answers[ref][0]
            rng, d = random.Random(7), []
            for i in range(args.boot):
                draw = [rng.choice(sel) for _ in sel]
                clips = [dict(c, clip=f"{c['clip']}#{j}") for j, c in enumerate(draw)]
                d.append(lift(clips, ans, 4, 100 + i)[0] - lift(clips, base, 4, 100 + i)[0])
            d.sort()
            vs[ref] = {"diff": pt - lift(sel, base, 20, 0)[0],
                       "ci95": [d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]]}
        ref8 = answers["8 frames"][0]
        keys = [k for k in ans if k in ref8]
        cams, died = cameras(live)
        rows.append({
            "arm": lab, "source": src, "tokens_per_call": tok, "lift": pt, "vs": vs,
            "false_alarms_per_hour": fa / hours,
            "same_answer_as_8_frames": sum(ans[k] == ref8[k] for k in keys) / len(keys),
            "same_something_nothing": sum(bool(ans[k]) == bool(ref8[k]) for k in keys) / len(keys),
            "cameras": cams, "server_died_at_cameras": died,
            "per_group": {g: {"n": v[1], "lift": v[0] / v[1] - ch_g.get(g, 0.0)}
                          for g, v in sorted(per.items()) if v[1]},
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"boot": args.boot, "hours": hours, "windows": sum(n_win.values()),
                        "refs": REFS}, "rows": rows}, open(args.out, "w"), indent=1)

    pct = lambda v: "    -" if v is None else f"{100 * v:+5.1f}"
    print(f"{'arm':42s} {'tok':>5s} {'lift':>6s} {'vs 8 frames':>22s} {'vs EVS 0.5':>22s} "
          f"{'FA/h':>6s} {'same':>6s} {'cams':>5s}")
    for r in rows:
        def v(ref):
            x = r["vs"].get(ref)
            return "" if x is None else f"{100*x['diff']:+.1f} [{100*x['ci95'][0]:+.1f},{100*x['ci95'][1]:+.1f}]"
        cams = "-" if r["cameras"] is None else str(r["cameras"])
        print(f"{r['arm']:42s} {r['tokens_per_call']:5.0f} {pct(r['lift'])} {v('8 frames'):>22s} "
              f"{v('8 frames, vLLM EVS 0.5'):>22s} {r['false_alarms_per_hour']:6.0f} "
              f"{100*r['same_answer_as_8_frames']:5.1f}% {cams:>5s}"
              + (f"  (server died at {r['server_died_at_cameras']})" if r["server_died_at_cameras"] else ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
