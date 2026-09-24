#!/usr/bin/env python3
"""E7e feasibility, before pre-registration: can group crops keep an action's actors
together, and what do they cost in vision tokens?

No VLM, no GPU. For every MEVA activity in the 24 E7 clips, on frames sampled once a
second inside it, the activity's annotated actors are matched to DeepStream's tracked
boxes (IoU >= 0.3, same class). Tracked people are then grouped by proximity:

  two people join a group if the gap between their boxes is <= k x the taller one's
  height; a vehicle joins a group if it is within k x that person's height.

An activity is "kept together" on a frame if every actor was tracked AND all of them
landed in one group -- i.e. one group crop would show the whole action. Per-person
crops keep an activity together only if it has a single actor.

Token cost per window (two frames, E7's crop rule: pad 25%, short side upscaled to
448 px up to 3x, then capped at 451,584 px; 28 px per token as in Qwen2.5-VL):
  full        the whole frame
  all-people  E7's ROI: one box around every tracked person (vehicles if no people)
  groups      one crop per group, all sent (summed)
  per-person  one crop per tracked person (summed)

    python3 experiments/e7e_groups.py
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import statistics as st
from pathlib import Path

from e7_gate import iou, load_geom, load_types

FW, FH = 1920, 1080
MAX_PX = 451_584
PATCH = 28
SAMPLE = 30                      # one frame a second
PERSON = {"person"}
VEHICLE = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}


def tokens(box) -> int:
    """Vision tokens for one image of this crop box under E7's crop rule."""
    if box is None:
        return 0
    x1, y1, x2, y2 = box
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if (w, h) != (FW, FH):
        k = min(3.0, max(1.0, 448 / min(w, h)))
        w, h = w * k, h * k
    if w * h > MAX_PX:
        s = math.sqrt(MAX_PX / (w * h))
        w, h = w * s, h * s
    return max(1, round(w / PATCH)) * max(1, round(h / PATCH))


def crop_box(boxes, pad=0.25, min_side=224):
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    bw, bh = x2 - x1, y2 - y1
    x1 -= pad * bw; x2 += pad * bw; y1 -= pad * bh; y2 += pad * bh
    if x2 - x1 < min_side:
        c = (x1 + x2) / 2; x1, x2 = c - min_side / 2, c + min_side / 2
    if y2 - y1 < min_side:
        c = (y1 + y2) / 2; y1, y2 = c - min_side / 2, c + min_side / 2
    return max(0, x1), max(0, y1), min(FW, x2), min(FH, y2)


def inside(box, crop, frac=0.5) -> bool:
    """At least half of the actor's box lies inside the crop."""
    ix = max(0.0, min(box[2], crop[2]) - max(box[0], crop[0]))
    iy = max(0.0, min(box[3], crop[3]) - max(box[1], crop[1]))
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    return ix * iy / area >= frac


def gap(a, b) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dy)


def groups(objs, k):
    """objs: [(cls, tid, box)]. Returns {tid: group id} for people and attached vehicles."""
    ppl = [o for o in objs if o[0] in PERSON]
    veh = [o for o in objs if o[0] in VEHICLE]
    parent = {o[1]: o[1] for o in ppl}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, a in enumerate(ppl):
        for b in ppl[i + 1:]:
            h = max(a[2][3] - a[2][1], b[2][3] - b[2][1])
            if gap(a[2], b[2]) <= k * h:
                parent[find(a[1])] = find(b[1])
    out = {o[1]: find(o[1]) for o in ppl}
    for v in veh:
        near = [(gap(v[2], p[2]), p[1]) for p in ppl
                if gap(v[2], p[2]) <= k * (p[2][3] - p[2][1])]
        if near:
            out[v[1]] = out[min(near)[1]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--repo", default="data/meva/meva-data-repo")
    ap.add_argument("--track-dir", default="data/meva/tracks")
    ap.add_argument("--ks", default="0.5,1,2")
    ap.add_argument("--out", default="results/e7e/groups_feasibility.json")
    args = ap.parse_args()
    ks = [float(x) for x in args.ks.split(",")]

    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    together = collections.defaultdict(lambda: collections.defaultdict(list))  # cls -> policy -> [frac]
    cost = collections.defaultdict(list)                                         # policy -> [tokens]
    ngroups = collections.defaultdict(list)
    n_act = collections.Counter()

    for c in sel:
        clip = c["clip"]
        meta = idx[clip]
        ann = Path(args.repo) / meta["annotation_dir"]
        geom, types = load_geom(ann / f"{clip}.geom.yml"), load_types(ann / f"{clip}.types.yml")
        trk = {}
        for line in gzip.open(Path(args.track_dir) / f"{clip}.jsonl.gz", "rt"):
            if line.strip():
                r = json.loads(line)
                trk[r["f"]] = [(o[0], o[1], tuple(o[3:7])) for o in r["o"]]

        # token cost, one sample per 2 s window (the VLM frame), frames with a person
        for f in range(18, meta["n_frames"], 60):
            objs = trk.get(f)
            if not objs or not any(o[0] in PERSON for o in objs):
                continue
            ppl = [o[2] for o in objs if o[0] in PERSON]
            cost["full"].append(2 * tokens((0, 0, FW, FH)))
            cost["all-people"].append(2 * tokens(crop_box(ppl)))
            cost["per-person"].append(sum(2 * tokens(crop_box([b])) for b in ppl))
            for k in ks:
                g = groups(objs, k)
                members = collections.defaultdict(list)
                for o in objs:
                    if o[1] in g:
                        members[g[o[1]]].append(o[2])
                cost[f"groups k={k:g}"].append(sum(2 * tokens(crop_box(m)) for m in members.values()))
                ngroups[k].append(len(members))

        for a in meta["activities"]:
            kinds = [types.get(t, "?") for t in a["actors"]]
            np_, nv = sum(k in PERSON for k in kinds), sum(k in VEHICLE for k in kinds)
            if np_ == 0:
                continue                                  # vehicle-only: out of scope for person crops
            cls = ("1 person" if np_ == 1 and nv == 0 else
                   "2+ people" if nv == 0 else "person + vehicle")
            n_act[cls] += 1
            fr = [f for f in range(a["start"] - a["start"] % 6, a["end"] + 1, 6) if f in trk]
            fr = fr[::5] or fr[:1]                       # ~1 per second
            hits = collections.defaultdict(list)
            for f in fr:
                objs = trk[f]
                matched = {}
                for t, kind in zip(a["actors"], kinds):
                    box = geom.get(t, {}).get(f)
                    if box is None:
                        continue
                    grp = PERSON if kind in PERSON else VEHICLE
                    best = max(((iou(o[2], box), o[1]) for o in objs if o[0] in grp
                                and o[1] not in matched.values()), default=(0, None))
                    if best[0] >= 0.3:
                        matched[t] = best[1]
                annotated = [t for t in a["actors"] if f in geom.get(t, {})]
                if not annotated:
                    continue
                found = all(t in matched for t in annotated)
                hits["tracked"].append(found)
                gt = [geom[t][f] for t in annotated]
                ppl = [o[2] for o in objs if o[0] in PERSON] or [o[2] for o in objs]
                hits["all-people"].append(found and all(inside(x, crop_box(ppl)) for x in gt))
                hits["per-person"].append(found and len(annotated) == 1)
                for k in ks:
                    g = groups(objs, k)
                    ids = {g.get(matched[t]) for t in annotated} if found else {None}
                    ok = found and None not in ids and len(ids) == 1
                    if ok:          # the group's crop must actually contain the actors
                        gid = ids.pop()
                        cb = crop_box([o[2] for o in objs if g.get(o[1]) == gid])
                        ok = all(inside(x, cb) for x in gt)
                    hits[f"groups k={k:g}"].append(ok)
            for pol, v in hits.items():
                together[cls][pol].append(sum(v) / len(v))

    res = {"activities": dict(n_act), "kept_together": {}, "tokens_per_window": {},
           "groups_per_window": {f"k={k:g}": st.mean(v) for k, v in ngroups.items()}}
    print(f"activities with a person actor: {dict(n_act)}")
    print("\nshare of frames where ONE crop shows every actor of the activity")
    pols = ["tracked", "all-people", "per-person"] + [f"groups k={k:g}" for k in ks]
    print(f"{'class':18s}" + "".join(f"{p:>14s}" for p in pols))
    for cls in ("1 person", "2+ people", "person + vehicle"):
        row = {p: st.mean(together[cls][p]) if together[cls][p] else None for p in pols}
        res["kept_together"][cls] = row
        print(f"{cls:18s}" + "".join(f"{100*row[p]:13.1f}%" if row[p] is not None else "             -"
                                     for p in pols))
    print("  ('tracked' = every actor found by the tracker: the ceiling for any crop policy;"
          " 'all-people' ignores vehicles when people are present)")
    print("\nvision tokens per window (2 frames), windows with a tracked person")
    for p, v in cost.items():
        v = sorted(v)
        res["tokens_per_window"][p] = {"mean": st.mean(v), "p50": v[len(v) // 2],
                                       "p90": v[int(0.9 * len(v))], "n": len(v)}
        print(f"  {p:14s} mean {st.mean(v):6.0f}  p50 {v[len(v)//2]:5d}  p90 {v[int(0.9*len(v))]:5d}")
    for k, v in ngroups.items():
        print(f"  groups k={k:g}: {st.mean(v):.2f} crops per window on average")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
