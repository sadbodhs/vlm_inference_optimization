#!/usr/bin/env python3
"""R1b: the label-free unit of work -- gate-fired windows and their proximity groups.

For every 2 s window of the 24 E7 clips, E7c's `track-motion` gate decides from
DeepStream's tracks whether the window goes to the VLM at all. In a fired window,
the people tracked on the two VLM frames (offsets 18 and 48) are grouped by
proximity -- two people join if the gap between their boxes is <= one body height;
a vehicle joins the nearest person within one body height -- exactly the rule of
the exploratory analysis (experiments/e7e_groups.py). No annotation is used to
choose a crop.

Annotations are used only to score: an activity is *covered* by a group if at
least half of one of its annotated actor boxes lies inside that group's 2x crop on
either VLM frame.

    python3 experiments/r1b_windows.py          # writes results/r1b/windows.json
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics as st
from pathlib import Path

from e7_gate import load_geom, load_types
from e7_vlm import TYPE_TO_GROUP, VLM_OFFSETS, WINDOW
from e7c import STRIDE, gate_window

FW, FH = 1920, 1080
PERSON = {"person"}
VEHICLE = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}
MAX_GROUPS = 7          # 2 frames x 7 groups + 1 context frame = 15 images <= 16


def gap(a, b):
    return max(max(0.0, max(a[0], b[0]) - min(a[2], b[2])),
               max(0.0, max(a[1], b[1]) - min(a[3], b[3])))


def union(bs):
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def expand(box, m=2.0):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = max(32.0, (box[2] - box[0]) * m), max(32.0, (box[3] - box[1]) * m)
    x1 = min(max(0, int(cx - w / 2)), FW - int(min(w, FW)))
    y1 = min(max(0, int(cy - h / 2)), FH - int(min(h, FH)))
    x1, y1 = max(0, x1), max(0, y1)
    return x1, y1, min(FW, x1 + int(w)), min(FH, y1 + int(h))


def inside(box, crop, frac=0.5):
    ix = max(0.0, min(box[2], crop[2]) - max(box[0], crop[0]))
    iy = max(0.0, min(box[3], crop[3]) - max(box[1], crop[1]))
    return ix * iy / max(1.0, (box[2] - box[0]) * (box[3] - box[1])) >= frac


def groups_for(objs_by_frame):
    """Proximity groups over the two VLM frames. objs: {frame: [(cls, tid, box)]}.
    A track is one node (its boxes on both frames); edges from either frame."""
    boxes = collections.defaultdict(list)
    cls = {}
    for f, objs in objs_by_frame.items():
        for c, tid, b in objs:
            boxes[tid].append(b); cls[tid] = c
    ppl = [t for t in boxes if cls[t] in PERSON]
    parent = {t: t for t in ppl}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, a in enumerate(ppl):
        for b in ppl[i + 1:]:
            if any(gap(x, y) <= max(x[3] - x[1], y[3] - y[1]) for x in boxes[a] for y in boxes[b]):
                parent[find(a)] = find(b)
    members = collections.defaultdict(list)
    for t in ppl:
        members[find(t)].append(t)
    for v in (t for t in boxes if cls[t] in VEHICLE):
        near = [(min(gap(x, y) for x in boxes[v] for y in boxes[p]), p) for p in ppl
                if any(gap(x, y) <= (y[3] - y[1]) for x in boxes[v] for y in boxes[p])]
        if near:
            members[find(min(near)[1])].append(v)
    out = [union([b for t in ts for b in boxes[t]]) for ts in members.values()]
    # largest first; beyond MAX_GROUPS the smallest are dropped (and counted)
    out.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--repo", default="data/meva/meva-data-repo")
    ap.add_argument("--track-dir", default="data/meva/tracks")
    ap.add_argument("--gate", default="track-motion")
    ap.add_argument("--out", default="results/r1b/windows.json")
    args = ap.parse_args()
    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}

    windows, acts_out = [], []
    n_all = dropped = 0
    for c in sel:
        clip, meta = c["clip"], idx[c["clip"]]
        ann = Path(args.repo) / meta["annotation_dir"]
        geom, types = load_geom(ann / f"{clip}.geom.yml"), load_types(ann / f"{clip}.types.yml")
        frames = {}
        for line in gzip.open(Path(args.track_dir) / f"{clip}.jsonl.gz", "rt"):
            if line.strip():
                r = json.loads(line); frames[r["f"]] = r["o"]
        n_win = meta["n_frames"] // WINDOW
        n_all += n_win
        fired = {}
        for w in range(n_win):
            win = [frames.get(fr, []) for fr in range(w * WINDOW, (w + 1) * WINDOW, STRIDE)]
            if not gate_window(args.gate, win, frames.get(w * WINDOW - STRIDE, [])):
                continue
            vf = [w * WINDOW + o for o in VLM_OFFSETS]
            objs = {f: [(o[0], o[1], tuple(o[3:7])) for o in frames.get(f, [])] for f in vf}
            gs = groups_for(objs)
            dropped += max(0, len(gs) - MAX_GROUPS)
            gs = gs[:MAX_GROUPS]
            crops = [expand(g) for g in gs]
            # truth: activities overlapping the window, and which crop (if any) holds them
            truth, covered = [], []
            for a in meta["activities"]:
                if not (a["start"] <= (w + 1) * WINDOW - 1 and a["end"] >= w * WINDOW):
                    continue
                g = TYPE_TO_GROUP.get(a["type"])
                if g is None:
                    continue
                truth.append(g)
                hold = sorted({i for i, cb in enumerate(crops) for t in a["actors"]
                               for f in vf if f in geom.get(t, {}) and inside(geom[t][f], cb)})
                covered.append({"act": a["id"], "group": g, "crops": hold})
            fired[w] = True
            windows.append({"id": f"{clip}|{w}", "clip": clip, "w": w, "frames": vf,
                            "bin": c["bin"], "crops": crops, "truth": sorted(set(truth)),
                            "covered": covered})
        for a in meta["activities"]:
            if a["type"] not in TYPE_TO_GROUP:
                continue
            ws = list(range(a["start"] // WINDOW, min(n_win - 1, a["end"] // WINDOW) + 1))
            acts_out.append({"clip": clip, "act": a["id"], "group": TYPE_TO_GROUP[a["type"]],
                             "windows": ws, "fired": [w for w in ws if w in fired],
                             "person": any(types.get(t) in PERSON for t in a["actors"])})

    fired_w = len(windows)
    ng = [len(w["crops"]) for w in windows]
    in_fired = [a for a in acts_out if a["fired"]]
    # an activity is crop-covered if some fired window holds one of its actors in a crop
    cov = collections.defaultdict(bool)
    for w in windows:
        for c in w["covered"]:
            if c["crops"]:
                cov[(w["clip"], c["act"])] = True
    person_fired = [a for a in in_fired if a["person"]]
    meta = {"gate": args.gate, "windows_all": n_all, "windows_fired": fired_w,
            "fired_frac": fired_w / n_all, "groups_per_fired_window_mean": st.mean(ng),
            "groups_per_fired_window_p90": sorted(ng)[int(0.9 * len(ng))],
            "fired_windows_without_groups": sum(n == 0 for n in ng), "groups_dropped_over_cap": dropped,
            "activities": len(acts_out), "activities_in_a_fired_window": len(in_fired),
            "person_activities_in_a_fired_window": len(person_fired),
            "person_activities_crop_covered": sum(cov[(a["clip"], a["act"])] for a in person_fired)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "windows": windows, "activities": acts_out}))
    for k, v in meta.items():
        print(f"  {k:40s} {v:.3f}" if isinstance(v, float) else f"  {k:40s} {v}")


if __name__ == "__main__":
    main()
