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
  tokens / call        mean prompt tokens of the arm (cost per call)
  cameras per 3090     usable capacity / (0.5 windows/s x call rate), when the
                       capacity sweep for that arm exists

    docker/run_harness.sh python3 experiments/e7_report.py --arm-id B0_vllm_awq_clean
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
    ap.add_argument("--arm-id", default="B0_vllm_awq_clean")
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
    cap = {}
    for which in ("full", "roi"):
        f = Path(args.results) / "sweeps" / f"e7-capacity-{which}-{args.arm_id}.json"
        if f.exists():
            cap[which] = json.loads(f.read_text())["meta"].get("usable_windows_per_s")

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
                    rows.append({
                        "size": s, "gate": g, "arm": which, "bin": b,
                        "call_rate": rate,
                        "gate_recall": gate_hit / inst if inst else None,
                        "recognised": recog / inst if inst else None,
                        "false_alarms_per_hour": fa / hours if hours else None,
                        "tokens_per_call": tok,
                        "cameras_per_gpu": (usable / (WINDOWS_PER_S * rate))
                                           if usable and rate else None,
                        "instances": inst,
                        "per_group": {k: v[0] / v[1] for k, v in per_group.items() if v[1]},
                    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"arm_id": args.arm_id, "capacity": cap, "sizes": sizes},
               "rows": rows}, open(args.out, "w"), indent=1)

    f = lambda v, p=True: "   -  " if v is None else (f"{100*v:5.1f}%" if p else f"{v:6.1f}")
    for s in sizes:
        print(f"\n=== YOLOv8s @ {s}, all clips ===")
        print(f"{'gate':14s} {'arm':5s} {'calls':>7s} {'gate rec':>9s} {'recog':>7s} "
              f"{'FA/h':>7s} {'tok/call':>9s} {'cams/GPU':>9s}")
        for r in rows:
            if r["size"] == s and r["bin"] == "all":
                cams = "   -" if r["cameras_per_gpu"] is None else f"{r['cameras_per_gpu']:9.1f}"
                tok = "   -" if r["tokens_per_call"] is None else f"{r['tokens_per_call']:9.0f}"
                print(f"{r['gate']:14s} {r['arm']:5s} {f(r['call_rate']):>7s} "
                      f"{f(r['gate_recall']):>9s} {f(r['recognised']):>7s} "
                      f"{f(r['false_alarms_per_hour'], False):>7s} {tok} {cams}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
