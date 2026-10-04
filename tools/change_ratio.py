#!/usr/bin/env python3
"""E9 step A (CPU only): how much of an 8-frame window actually changes?

Before spending GPU time on pruning patches ahead of the vision encoder, measure
the most it could save on the E7 clips, and what it would throw away.

Per window, the 8 frames E8 sent as video (offsets 6, 12, ..., 48) are decoded at
the size the model sees (<= 451,584 px, 896 x 504 for 1080p) and grouped into the
4 frame pairs Qwen merges. Pair 0 is the keyframe and is always kept whole. Pairs
1-3 keep only the 28 px blocks (one LLM token each) that changed against the
previous pair:

  pixel   grayscale, Gaussian blur, |frame - same frame of the previous pair|,
          max over the pair's 2 frames, mean per block > threshold (grey levels),
          optionally dilated by 1 block
  tracker blocks under a DeepStream/NvDCF track box on either frame of the pair,
          dilated by 1 block (the boxes the live pipeline already has)

Reported per detector and threshold: share of pair-1..3 blocks kept, estimated
visual tokens per request (keyframe + kept blocks), share of windows whose later
pairs keep nothing, and ACTOR RECALL -- of the MEVA-annotated actors of activities
under way in the window, the share of their box blocks the mask keeps (pairs 1-3;
the keyframe always shows them).

    docker run --rm --cpus 6 -u 1000:1000 -v $PWD:/work -w /work vlmbench-harness:latest \\
        python3 tools/change_ratio.py
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from e7_gate import load_geom, load_types  # noqa: E402

WINDOW, W, H, BLOCK = 60, 896, 504, 28
OFFSETS = tuple(range(6, 49, 6))                      # E8's vid8 frames
GW, GH = W // BLOCK, H // BLOCK                       # 32 x 18 blocks = 576 tokens per pair
SRC_W, SRC_H = 1920, 1080
THRESHOLDS = (2, 3, 4, 6, 8, 12, 16)


def gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L").resize((W, H), Image.BILINEAR)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32)


def block_mean(a: np.ndarray) -> np.ndarray:
    return a.reshape(GH, BLOCK, GW, BLOCK).mean(axis=(1, 3))


def dilate(m: np.ndarray) -> np.ndarray:
    p = np.pad(m, 1)
    return np.max([p[i:i + GH, j:j + GW] for i in range(3) for j in range(3)], axis=0)


def box_mask(boxes) -> np.ndarray:
    m = np.zeros((GH, GW), bool)
    for x1, y1, x2, y2 in boxes:
        c1 = max(0, int(x1 * W / SRC_W // BLOCK)); c2 = min(GW - 1, int(x2 * W / SRC_W // BLOCK))
        r1 = max(0, int(y1 * H / SRC_H // BLOCK)); r2 = min(GH - 1, int(y2 * H / SRC_H // BLOCK))
        if c2 >= c1 and r2 >= r1:
            m[r1:r2 + 1, c1:c2 + 1] = True
    return m


def one_clip(job):
    clip, frame_dir, track_dir, ann_dir, acts, n_frames = job
    tracks = {}
    with gzip.open(Path(track_dir) / f"{clip}.jsonl.gz", "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                tracks[r["f"]] = [o[3:7] for o in r["o"]]
    geom = load_geom(Path(ann_dir) / f"{clip}.geom.yml")
    rows = []
    for w in range(n_frames // WINDOW):
        base = w * WINDOW
        fr = [base + o for o in OFFSETS]
        g = [gray(Path(frame_dir) / clip / f"{f:05d}.jpg") for f in fr]
        # pair k = frames (2k, 2k+1); change vs the same frame of pair k-1
        score = [block_mean(np.maximum(np.abs(g[2 * k] - g[2 * k - 2]),
                                       np.abs(g[2 * k + 1] - g[2 * k - 1]))) for k in (1, 2, 3)]
        trk = [dilate(box_mask(tracks.get(fr[2 * k], []) + tracks.get(fr[2 * k + 1], [])))
               for k in (1, 2, 3)]
        # actors of activities under way, per later pair (annotation boxes, every frame)
        actor = []
        for k in (1, 2, 3):
            boxes = []
            for a in acts:
                if a["end"] < fr[2 * k] or a["start"] > fr[2 * k + 1]:
                    continue
                for tid in a["actors"]:
                    for f in (fr[2 * k], fr[2 * k + 1]):
                        if f in geom.get(tid, {}):
                            boxes.append(geom[tid][f])
            actor.append([box_mask([b]) for b in boxes])
        row = {"w": w, "active": any(a["start"] <= base + WINDOW - 1 and a["end"] >= base for a in acts),
               "keep": {}, "recall": {}}

        def record(name, masks):
            row["keep"][name] = float(np.mean([m.mean() for m in masks]))
            hit = tot = 0
            for m, am in zip(masks, actor):
                for b in am:
                    tot += b.sum(); hit += (b & m).sum()
            row["recall"][name] = (int(hit), int(tot))

        for t in THRESHOLDS:
            raw = [s > t for s in score]
            record(f"pixel>{t}", raw)
            record(f"pixel>{t}+dil", [dilate(m) for m in raw])
        record("tracker", trk)
        rows.append(row)
    return clip, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--frame-dir", default="data/meva/frames5")
    ap.add_argument("--track-dir", default="data/meva/tracks")
    ap.add_argument("--ann-root", default="data/meva/meva-data-repo")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument("--out", default="results/e9/change_ratio.json")
    args = ap.parse_args()

    sel = json.load(open(args.selection))["clips"]
    if args.limit_clips:
        sel = sel[: args.limit_clips]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    jobs = [(c["clip"], args.frame_dir, args.track_dir,
             str(Path(args.ann_root) / idx[c["clip"]]["annotation_dir"]),
             idx[c["clip"]]["activities"], idx[c["clip"]]["n_frames"]) for c in sel]
    with Pool(args.procs) as pool:
        per_clip = dict(pool.map(one_clip, jobs))

    bins = {c["clip"]: c["bin"] for c in sel}
    names = list(next(iter(per_clip.values()))[0]["keep"])
    summary = {}
    for b in ("empty", "sparse", "moderate", "busy", "all"):
        rows = [r for cl, rs in per_clip.items() for r in rs if b == "all" or bins[cl] == b]
        s = {}
        for n in names:
            keep = [r["keep"][n] for r in rows]
            hit = sum(r["recall"][n][0] for r in rows); tot = sum(r["recall"][n][1] for r in rows)
            s[n] = {"keep": float(np.mean(keep)),
                    "tokens_vs_vid8": (1 + 3 * float(np.mean(keep))) / 4,
                    "windows_nothing_kept": float(np.mean([k == 0 for k in keep])),
                    "active_windows_nothing_kept": float(np.mean([r["keep"][n] == 0 for r in rows
                                                                  if r["active"]] or [0])),
                    "actor_recall": hit / tot if tot else None}
        summary[b] = {"windows": len(rows), "by_detector": s}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"offsets": OFFSETS, "size": [W, H], "block": BLOCK,
                        "note": "pair 0 always kept; keep = share of pair 1-3 blocks; "
                                "tokens_vs_vid8 = (576 + kept) / 2304 visual tokens"},
               "summary": summary, "per_clip": per_clip}, open(args.out, "w"))

    a = summary["all"]["by_detector"]
    print(f"{summary['all']['windows']} windows, {len(per_clip)} clips")
    print(f"{'detector':16s} {'kept':>6s} {'tokens vs vid8':>15s} {'nothing kept':>13s} "
          f"{'active & nothing':>17s} {'actor recall':>13s}")
    for n in names:
        r = a[n]
        print(f"{n:16s} {100*r['keep']:5.1f}% {100*r['tokens_vs_vid8']:14.1f}% "
              f"{100*r['windows_nothing_kept']:12.1f}% {100*r['active_windows_nothing_kept']:16.1f}% "
              + ("            -" if r["actor_recall"] is None else f"{100 * r['actor_recall']:12.1f}%"))
    print("\nkept share by duty-cycle bin (pixel>4+dil, tracker):")
    for b in ("empty", "sparse", "moderate", "busy"):
        s = summary[b]["by_detector"]
        print(f"  {b:9s} {100*s['pixel>4+dil']['keep']:5.1f}%  {100*s['tracker']['keep']:5.1f}%")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
