#!/usr/bin/env python3
"""E7 stage 2: replay every gate over the detections and price it against MEVA.

No GPU, no VLM: a gate's decision is a function of the detections alone, so every
policy can be evaluated exactly, on identical inputs, from one detection pass.

For each gate and detector input size:
  call rate        fraction of 2 s windows the gate sends to the VLM
  activity recall  fraction of annotated activity instances overlapped by >= 1
                   fired window -- an instance no fired window touches is one the
                   VLM can never report, however good it is
Both overall and per duty-cycle bin, because savings are a property of the scene.

Also, independent of any gate: detector recall on the annotated actors, by the
actor's box height -- the mechanism behind whatever recall the gates lose.

    docker/run_harness.sh python3 experiments/e7_gate.py
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path

WINDOW, STRIDE = 60, 6
PERSON = {"person"}
VEHICLE = {"car", "truck", "bus", "motorcycle", "bicycle"}
GATES = ("dense", "presence", "person", "motion", "person-motion")
HEIGHT_BINS = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 10_000)]
IOU_MATCH = 0.3        # detector-vs-annotation match, loose: small boxes are jittery
MOVE_FRAC = 0.1        # "moved": centre displacement > 0.1 x box height


# ── annotations ──────────────────────────────────────────────────────────────
# MEVA ships KPF in two layouts, sometimes within one date directory:
#   - { geom: {id1: 561..., id0: 1, ts0: 942, ts1: 31.4, g0: 0 548 36 635, ...} }
#   - {'geom': {'g0': '1578 321 1650 488', 'id0': 1, 'id1': 39, 'ts0': 8483}}
# Different key order, different quoting. Each field is matched on its own, and a
# file with geometry lines that yields no tracks is an error -- a parser that
# silently returns nothing is how the first run's detector-recall table came out
# blank without anything failing.
_id1 = re.compile(r"'?id1'?:\s*(\d+)")
_ts0 = re.compile(r"'?ts0'?:\s*(\d+)")
_g0 = re.compile(r"'?g0'?:\s*'?(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)")
_cset = re.compile(r"'?cset3'?:\s*\{\s*'?(\w+)'?\s*:")


def load_geom(path: Path) -> dict[int, dict[int, tuple]]:
    """{track id: {frame: (x1, y1, x2, y2)}} from a KPF geom.yml, either layout."""
    tracks: dict[int, dict[int, tuple]] = collections.defaultdict(dict)
    n_geom = 0
    with open(path) as f:
        for line in f:
            if "geom" not in line:
                continue
            n_geom += 1
            i, t, g = _id1.search(line), _ts0.search(line), _g0.search(line)
            if i and t and g:
                tracks[int(i.group(1))][int(t.group(1))] = tuple(map(int, g.groups()))
    if n_geom and not tracks:
        raise ValueError(f"{path}: {n_geom} geom lines, none parsed -- unknown KPF layout")
    return tracks


def load_types(path: Path) -> dict[int, str]:
    out, n = {}, 0
    with open(path) as f:
        for line in f:
            if "types" not in line:
                continue
            n += 1
            i, c = _id1.search(line), _cset.search(line)
            if i and c:
                out[int(i.group(1))] = c.group(1).lower()
    if n and not out:
        raise ValueError(f"{path}: {n} types lines, none parsed -- unknown KPF layout")
    return out


# ── detections ───────────────────────────────────────────────────────────────
def load_dets(path: Path) -> dict[int, list]:
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            out[r["f"]] = r["d"]          # [cls, conf, x1, y1, x2, y2]
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def changed(prev: list, cur: list, groups: set) -> bool:
    """Did anything in `groups` appear, vanish, or move between two sampled frames?

    Greedy IoU matching within a class group. No tracker, by design (PLAN.md 10):
    this is the cheapest motion signal, and its flicker is part of what it costs.
    """
    p = [d for d in prev if d[0] in groups]
    c = [d for d in cur if d[0] in groups]
    if len(p) != len(c):
        return True
    used = set()
    for a in p:
        best, bi = 0.0, -1
        for j, b in enumerate(c):
            if j in used:
                continue
            same = (a[0] in PERSON) == (b[0] in PERSON)
            v = iou(a[2:], b[2:]) if same else 0.0
            if v > best:
                best, bi = v, j
        if bi < 0 or best < 0.1:
            return True
        used.add(bi)
        b = c[bi]
        h = max(a[5] - a[3], b[5] - b[3], 1.0)
        dx = (a[2] + a[4]) / 2 - (b[2] + b[4]) / 2
        dy = (a[3] + a[5]) / 2 - (b[3] + b[5]) / 2
        if (dx * dx + dy * dy) ** 0.5 > MOVE_FRAC * h:
            return True
    return False


def fire(dets: dict[int, list], n_frames: int) -> dict[str, list[bool]]:
    n_win = n_frames // WINDOW
    out = {g: [False] * n_win for g in GATES}
    for w in range(n_win):
        frames = [f for f in range(w * WINDOW, (w + 1) * WINDOW, STRIDE)]
        ds = [dets.get(f, []) for f in frames]
        prev = dets.get(frames[0] - STRIDE, [])
        out["dense"][w] = True
        out["presence"][w] = any(d[0] in PERSON | VEHICLE for fd in ds for d in fd)
        out["person"][w] = any(d[0] in PERSON for fd in ds for d in fd)
        seq = [prev] + ds
        out["motion"][w] = any(changed(a, b, PERSON | VEHICLE) for a, b in zip(seq, seq[1:]))
        out["person-motion"][w] = any(changed(a, b, PERSON) for a, b in zip(seq, seq[1:]))
    return out


def actor_height(act: dict, tracks: dict) -> float | None:
    hs = [b[3] - b[1] for a in act["actors"] for fr, b in tracks.get(a, {}).items()
          if act["start"] <= fr <= act["end"]]
    return sorted(hs)[len(hs) // 2] if hs else None


def hbin(h: float) -> str:
    for lo, hi in HEIGHT_BINS:
        if lo <= h < hi:
            return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--ann-root", default="data/meva/meva-data-repo")
    ap.add_argument("--det-dir", default="data/meva/det")
    ap.add_argument("--sizes", default="640,1280")
    ap.add_argument("--out", default="results/e7/gates.json")
    args = ap.parse_args()

    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    sizes = [int(s) for s in args.sizes.split(",")]

    # per (size, gate, bin): windows fired / total; activities recalled / total
    calls = collections.defaultdict(lambda: [0, 0])
    recall = collections.defaultdict(lambda: [0, 0])
    by_height = collections.defaultdict(lambda: [0, 0])     # (size, gate, hbin)
    by_type = collections.defaultdict(lambda: [0, 0])       # (size, gate, type)
    det_recall = collections.defaultdict(lambda: [0, 0])    # (size, kind, hbin)
    per_clip = []

    for c in sel:
        clip, b = c["clip"], c["bin"]
        meta = idx[clip]
        n_frames = meta["n_frames"]
        ann_dir = Path(args.ann_root) / meta["annotation_dir"]
        tracks = load_geom(ann_dir / f"{clip}.geom.yml")
        types = load_types(ann_dir / f"{clip}.types.yml")
        acts = meta["activities"]
        heights = [actor_height(a, tracks) for a in acts]

        for s in sizes:
            dets = load_dets(Path(args.det_dir) / f"{clip}.{s}.jsonl.gz")
            fired = fire(dets, n_frames)
            row = {"clip": clip, "bin": b, "size": s}
            for g in GATES:
                fw = fired[g]
                for key in ((s, g, b), (s, g, "all")):
                    calls[key][0] += sum(fw)
                    calls[key][1] += len(fw)
                hit_n = 0
                for a, h in zip(acts, heights):
                    ws = range(max(0, a["start"] // WINDOW), min(len(fw) - 1, a["end"] // WINDOW) + 1)
                    hit = any(fw[w] for w in ws)
                    hit_n += hit
                    for key in ((s, g, b), (s, g, "all")):
                        recall[key][0] += hit
                        recall[key][1] += 1
                    by_type[(s, g, a["type"])][0] += hit
                    by_type[(s, g, a["type"])][1] += 1
                    if h is not None:
                        by_height[(s, g, hbin(h))][0] += hit
                        by_height[(s, g, hbin(h))][1] += 1
                row[g] = {"call_rate": sum(fw) / len(fw),
                          "recall": hit_n / len(acts) if acts else None}
            per_clip.append(row)

            # detector recall on annotated actors, at the sampled frames of their activities
            for a in acts:
                for tid in a["actors"]:
                    kind = types.get(tid, "other")
                    groups = PERSON if kind == "person" else VEHICLE if kind == "vehicle" else None
                    if groups is None:
                        continue
                    for fr, box in tracks.get(tid, {}).items():
                        if fr % STRIDE or not (a["start"] <= fr <= a["end"]):
                            continue
                        found = any(d[0] in groups and iou(d[2:], box) >= IOU_MATCH
                                    for d in dets.get(fr, []))
                        k = (s, kind, hbin(box[3] - box[1]))
                        det_recall[k][0] += found
                        det_recall[k][1] += 1
        print(f"  {clip} [{b}] done")

    def table(d):
        return {"|".join(map(str, k)): {"hit": v[0], "n": v[1],
                                         "rate": (v[0] / v[1]) if v[1] else None}
                for k, v in sorted(d.items(), key=lambda kv: tuple(map(str, kv[0])))}

    out = {"meta": {"window_frames": WINDOW, "stride": STRIDE, "iou_match": IOU_MATCH,
                    "move_frac": MOVE_FRAC, "sizes": sizes, "gates": GATES},
           "calls": table(calls), "recall": table(recall), "recall_by_height": table(by_height),
           "recall_by_type": table(by_type), "detector_recall": table(det_recall),
           "per_clip": per_clip}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    bins = ["empty", "sparse", "moderate", "busy", "all"]
    for s in sizes:
        print(f"\n=== YOLOv8s @ {s} ===   call rate / activity recall")
        print(f"{'gate':14s}" + "".join(f"{b:>18s}" for b in bins))
        for g in GATES:
            cells = []
            for b in bins:
                cr = calls[(s, g, b)]
                rc = recall[(s, g, b)]
                c_ = f"{100*cr[0]/cr[1]:5.1f}%" if cr[1] else "   - "
                r_ = f"{100*rc[0]/rc[1]:5.1f}%" if rc[1] else "   - "
                cells.append(f"{c_} / {r_}".rjust(18))
            print(f"{g:14s}" + "".join(cells))
        print("  detector recall on annotated actors, by box height (px):")
        for kind in ("person", "vehicle"):
            cells = []
            for lo, hi in HEIGHT_BINS:
                k = (s, kind, f"{lo}-{hi}" if hi < 10_000 else f"{lo}+")
                v = det_recall.get(k, [0, 0])
                cells.append(f"{k[2]:>8s}: {100*v[0]/v[1]:5.1f}% (n={v[1]})" if v[1] else f"{k[2]:>8s}: -")
            print(f"    {kind:8s}" + "  ".join(cells))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
