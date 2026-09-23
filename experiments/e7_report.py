#!/usr/bin/env python3
"""E7 stage 5: replay every cascade = (detector size, gate, VLM arm) and price it.

For a window the gate fires on, the cascade's answer is that arm's answer for the
window; for a window it skips, the answer is "nothing happening". Exact, because a
gate changes whether a window is sent, not what is sent (see e7_vlm.py).

Per cascade:
  call rate            windows sent / windows
  gate recall          activity instances with >= 1 sent window (the ceiling)
  recognised           instances whose group letter appears in a sent window's answer
  false alarms / hour  letters asserted with no annotated activity of that group
                       overlapping the window, per camera-hour
  recognised (chance)  the same, with every sent window's answer replaced by a
                       random other sent window's answer (20 shuffles). Keeps the
                       arm's answer mix and the gate's call pattern, breaks the
                       link to the frame: an arm that asserts more letters, or a
                       long activity spanning many windows, scores here by luck
  lift                 recognised - chance: what the VLM actually saw
  tokens / call        mean prompt tokens of the arm (cost per call)
  cameras per 3090     1 / (detector share + VLM share) per camera, where
                       detector share = 5 fps x measured ms/frame (0 for dense,
                       which runs no detector) and VLM share = 0.5 windows/s x
                       call rate / usable capacity. A time-sharing MODEL of one
                       GPU running both -- not a co-hosted measurement

    docker/run_harness.sh python3 experiments/e7_report.py --arm-id V_vllm_video
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from e7_gate import GATES, fire, load_dets  # noqa: E402
from e7_vlm import GROUPS, TYPE_TO_GROUP, WINDOW  # noqa: E402

LETTERS = set(GROUPS)
WINDOWS_PER_S = 30 / WINDOW          # one decision per 2 s window


def parse_answer(text: str) -> set[str]:
    """Letters A-H the model asserted. 'N' / 'none' -> empty set.

    Only the LEADING run of single-letter tokens counts ("B, D", "B and D"), so a
    sentence answer -- "A person walks through..." -- is not read as option A.
    """
    toks = re.findall(r"[A-Za-z]+", (text or "").strip())
    if not toks or toks[0].upper() in ("N", "NONE", "NO"):
        return set()
    out = set()
    for t in toks:
        u = t.upper()
        if len(t) == 1 and u in LETTERS:
            out.add(u)
        elif u == "AND":
            continue
        else:
            break
    # a lone leading "A" followed by prose is an article, not an answer
    if out == {"A"} and len(toks) > 2 and toks[1].lower() not in ("and",):
        return set()
    return out


def recognised(sel_clips, idx, fired, answer_of):
    """(hits, instances, {group: [hits, n]}) for one answer assignment."""
    hit = n = 0
    per = collections.defaultdict(lambda: [0, 0])
    for c in sel_clips:
        clip = c["clip"]
        fw = fired[clip]
        for a in idx[clip]["activities"]:
            grp = TYPE_TO_GROUP[a["type"]]
            ws = range(a["start"] // WINDOW, min(len(fw) - 1, a["end"] // WINDOW) + 1)
            h = any(fw[w] and grp in answer_of(clip, w) for w in ws)
            n += 1
            hit += h
            per[grp][0] += h
            per[grp][1] += 1
    return hit, n, per


def chance_recognised(sel_clips, idx, fired, dense_answer_of, k=20, seed=0):
    """Recognition when the ARM'S DENSE answers are shuffled across all windows,
    then the gate's mask applied. One shuffle pool for every gate, so a gate's
    chance baseline differs from dense only by what the gate removes -- shuffling
    among each gate's own sent windows would give every gate its own, inflated
    baseline (sent windows lean active) and make lifts incomparable across gates.
    Returns (overall rate, {group: rate})."""
    import random
    allw = [(c["clip"], w) for c in sel_clips for w in range(len(fired[c["clip"]]))]
    pool = [dense_answer_of(cl, w) for cl, w in allw]
    rng = random.Random(seed)
    tot, per_tot = [], collections.defaultdict(list)
    for _ in range(k):
        perm = pool[:]
        rng.shuffle(perm)
        m = dict(zip(allw, perm))
        h, n, per = recognised(sel_clips, idx, fired,
                               lambda cl, w: m[(cl, w)] if fired[cl][w] else set())
        tot.append(h / n if n else 0.0)
        for g, (hh, nn) in per.items():
            per_tot[g].append(hh / nn)
    return sum(tot) / len(tot), {g: sum(v) / len(v) for g, v in per_tot.items()}


def bootstrap_lift(sel_clips, idx, fired, answer_of, dense_answer_of, b=300, k=4, seed=1):
    """95% CI on lift, resampling CLIPS -- activities within a clip share a camera,
    a scene and often a crowd, so they are not independent samples."""
    import random
    rng = random.Random(seed)
    out = []
    for i in range(b):
        res = [rng.choice(sel_clips) for _ in sel_clips]
        # duplicate clips need distinct keys: rename per draw
        clips = [dict(c, clip=f"{c['clip']}#{j}") for j, c in enumerate(res)]
        idx2 = {c["clip"]: idx[c["clip"].split("#")[0]] for c in clips}
        fired2 = {c["clip"]: fired[c["clip"].split("#")[0]] for c in clips}
        a2 = lambda cl, w: answer_of(cl.split("#")[0], w)
        d2 = lambda cl, w: dense_answer_of(cl.split("#")[0], w)
        h, n, _ = recognised(clips, idx2, fired2, a2)
        ch, _ = chance_recognised(clips, idx2, fired2, d2, k=k, seed=seed + i)
        if n:
            out.append(h / n - ch)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]


def paired_loss(sel_clips, idx, fired_gate, answer_gate, answer_dense, b=1000, seed=2):
    """Recognitions a gate loses vs dense -- SAME arm, SAME answers, so the only
    difference is the windows the gate skipped. Paired over clips: the between-
    clip variance that makes the VLM's own skill uncertain cancels here.
    Returns (point, lo, hi) in percentage points of instances."""
    import random
    per_clip = []
    for c in sel_clips:
        one = [c]
        dense_fired = {c["clip"]: [True] * len(fired_gate[c["clip"]])}
        hd, n, _ = recognised(one, idx, dense_fired, answer_dense)
        hg, _, _ = recognised(one, idx, fired_gate, answer_gate)
        per_clip.append((hd, hg, n))
    def delta(rows):
        n = sum(r[2] for r in rows)
        return 100 * (sum(r[1] for r in rows) - sum(r[0] for r in rows)) / n if n else 0.0
    rng = random.Random(seed)
    boots = sorted(delta([rng.choice(per_clip) for _ in per_clip]) for _ in range(b))
    return delta(per_clip), boots[int(0.025 * b)], boots[int(0.975 * b) - 1]


def load_answers(run_dir: Path) -> tuple[dict[str, set], float | None]:
    ans = {}
    if not (run_dir / "outputs.jsonl").exists():
        return ans, None
    for line in (run_dir / "outputs.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["ok"]:
                ans[r["sample_id"]] = parse_answer(r["text"])
    s = json.loads((run_dir / "summary.json").read_text())
    return ans, s.get("prompt_tokens", {}).get("mean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--det-dir", default="data/meva/det")
    ap.add_argument("--results", default="results")
    ap.add_argument("--arm-id", default="V_vllm_video")
    ap.add_argument("--sizes", default="640,1280")
    ap.add_argument("--out", default="results/e7/cascades.json")
    args = ap.parse_args()

    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    sizes = [int(s) for s in args.sizes.split(",")]
    arms = {}
    for which in ("full", "roi"):
        a, tok = load_answers(Path(args.results) / f"e7-vlm-{which}-{args.arm_id}")
        if a:
            arms[which] = (a, tok)
    if "full" not in arms:
        raise SystemExit("no full-frame VLM run found; run e7_vlm.py first")
    # Usable capacity: the highest achieved rate whose TTFT p99 is inside the 2 s
    # freshness budget. Deviation from PLAN.md 10, recorded on the page: the
    # pre-registered rule also required the harness's drift verdict to read
    # "not-saturated", and on these two-image inputs that verdict flagged nearly
    # every rate -- including ROI at 1.5/s with a 400 ms median, identical to 2.0/s.
    # The 160-request sweep is used; rates below its range come from the 80-request
    # sweep, which is the only measurement there.
    cap = {}
    for which in ("full", "roi"):
        sw = Path(args.results) / "sweeps"
        main = sw / f"e7-capacity-{which}-{args.arm_id}.json"
        low = sw / f"e7-capacity-{which}-{args.arm_id}-n80.json"
        rows_ = []
        if main.exists():
            rows_ = json.loads(main.read_text())["rows"]
            lo_rate = min(r["offered_qps"] for r in rows_)
            if low.exists():
                rows_ += [r for r in json.loads(low.read_text())["rows"]
                          if r["offered_qps"] < lo_rate]
        fresh = [r["achieved_qps"] for r in rows_ if r["ttft_p99_ms"] <= 2000]
        if fresh:
            cap[which] = max(fresh)

    det_ms = {}
    dc = Path("results/e7/detect_cost.json")
    if dc.exists():
        det_ms = {int(k): v["ms_p50"] for k, v in json.loads(dc.read_text())["cost_ms"].items()}

    rows = []
    for s in sizes:
        fired_by_clip = {c["clip"]: fire(load_dets(Path(args.det_dir) / f"{c['clip']}.{s}.jsonl.gz"),
                                         idx[c["clip"]]["n_frames"]) for c in sel}
        for g in GATES:
            for which, (answers, tok) in arms.items():
                for b in ("empty", "sparse", "moderate", "busy", "all"):
                    clips = [c for c in sel if b == "all" or c["bin"] == b]
                    n_win = sent = inst = gate_hit = recog = fa = 0
                    per_group = collections.defaultdict(lambda: [0, 0])
                    for c in clips:
                        clip = c["clip"]
                        fw = fired_by_clip[clip][g]
                        n_win += len(fw)
                        sent += sum(fw)

                        def answer(w):
                            if not fw[w]:
                                return set()
                            a = answers.get(f"{clip}|{w}")
                            if a is None and which == "roi":   # nothing to crop: full frame
                                a = arms["full"][0].get(f"{clip}|{w}", set())
                            return a or set()

                        acts = idx[clip]["activities"]
                        for w in range(len(fw)):
                            if not fw[w]:
                                continue
                            lo, hi = w * WINDOW, (w + 1) * WINDOW - 1
                            truth = {TYPE_TO_GROUP[a["type"]] for a in acts
                                     if a["start"] <= hi and a["end"] >= lo}
                            fa += len(answer(w) - truth)
                        for a in acts:
                            grp = TYPE_TO_GROUP[a["type"]]
                            ws = range(a["start"] // WINDOW, min(len(fw) - 1, a["end"] // WINDOW) + 1)
                            inst += 1
                            gh = any(fw[w] for w in ws)
                            rc = any(grp in answer(w) for w in ws)
                            gate_hit += gh
                            recog += rc
                            per_group[grp][0] += rc
                            per_group[grp][1] += 1
                    hours = n_win * WINDOW / 30 / 3600
                    rate = sent / n_win if n_win else 0
                    usable = cap.get(which)
                    fired_g = {c["clip"]: fired_by_clip[c["clip"]][g] for c in clips}

                    def answer_of(cl, w, _fired=fired_g, _answers=answers, _which=which):
                        if not _fired[cl][w]:
                            return set()
                        a = _answers.get(f"{cl}|{w}")
                        if a is None and _which == "roi":
                            a = arms["full"][0].get(f"{cl}|{w}", set())
                        return a or set()

                    def dense_answer_of(cl, w, _answers=answers, _which=which):
                        a = _answers.get(f"{cl}|{w}")
                        if a is None and _which == "roi":
                            a = arms["full"][0].get(f"{cl}|{w}", set())
                        return a or set()

                    chance, chance_group = chance_recognised(clips, idx, fired_g, dense_answer_of)
                    ci = (bootstrap_lift(clips, idx, fired_g, answer_of, dense_answer_of)
                          if b == "all" and s == sizes[-1] and g == "dense" else None)
                    loss = (paired_loss(clips, idx, fired_g, answer_of, dense_answer_of)
                            if b == "all" and g != "dense" else None)
                    det_share = 0.0 if g == "dense" else 5 * det_ms.get(s, 0.0) / 1000
                    vlm_share = (WINDOWS_PER_S * rate / usable) if usable else None
                    rows.append({
                        "size": s, "gate": g, "arm": which, "bin": b,
                        "call_rate": rate,
                        "gate_recall": gate_hit / inst if inst else None,
                        "recognised": recog / inst if inst else None,
                        "recognised_chance": chance,
                        "lift": (recog / inst - chance) if inst else None,
                        "lift_ci95": ci,
                        "recog_change_vs_dense_pts": loss,
                        "detector_share": det_share,
                        "vlm_share": vlm_share,
                        "false_alarms_per_hour": fa / hours if hours else None,
                        "tokens_per_call": tok,
                        "cameras_per_gpu_vlm_only": (usable / (WINDOWS_PER_S * rate))
                                                    if usable and rate else None,
                        "cameras_per_gpu": (1 / (det_share + vlm_share))
                                           if vlm_share else None,
                        "instances": inst,
                        "per_group": {k: {"n": v[1], "recognised": v[0] / v[1],
                                          "chance": chance_group.get(k),
                                          "lift": v[0] / v[1] - chance_group.get(k, 0.0)}
                                      for k, v in per_group.items() if v[1]},
                    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"arm_id": args.arm_id, "capacity": cap, "sizes": sizes},
               "rows": rows}, open(args.out, "w"), indent=1)

    f = lambda v, p=True: "   -  " if v is None else (f"{100*v:5.1f}%" if p else f"{v:6.1f}")
    for s in sizes:
        print(f"\n=== YOLOv8s @ {s}, all clips ===")
        print(f"{'gate':14s} {'arm':5s} {'calls':>7s} {'gate rec':>9s} {'recog':>7s} "
              f"{'chance':>7s} {'lift':>7s} {'FA/h':>7s} {'tok/call':>9s} {'cams/GPU':>9s}")
        for r in rows:
            if r["size"] == s and r["bin"] == "all":
                cams = "   -" if r["cameras_per_gpu"] is None else f"{r['cameras_per_gpu']:9.1f}"
                tok = "   -" if r["tokens_per_call"] is None else f"{r['tokens_per_call']:9.0f}"
                ci, lo = r.get("lift_ci95"), r.get("recog_change_vs_dense_pts")
                cis = (f"  skill CI [{100*ci[0]:+.1f},{100*ci[1]:+.1f}]" if ci else
                       f"  vs dense {lo[0]:+.1f} pts [{lo[1]:+.1f},{lo[2]:+.1f}]" if lo else "")
                print(f"{r['gate']:14s} {r['arm']:5s} {f(r['call_rate']):>7s} "
                      f"{f(r['gate_recall']):>9s} {f(r['recognised']):>7s} "
                      f"{f(r['recognised_chance']):>7s} {f(r['lift']):>7s} "
                      f"{f(r['false_alarms_per_hour'], False):>7s} {tok} {cams}{cis}")
    print("\n=== per activity group: dense, YOLO 1280 (recognised / chance / lift, n) ===")
    for r in rows:
        if r["size"] == sizes[-1] and r["gate"] == "dense" and r["bin"] == "all":
            print(f"  [{r['arm']}]")
            for g_, v in sorted(r["per_group"].items()):
                print(f"    {g_} {GROUPS[g_][0][:52]:52s} {100*v['recognised']:5.1f}% "
                      f"{100*(v['chance'] or 0):5.1f}% {100*v['lift']:+6.1f}  n={v['n']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
