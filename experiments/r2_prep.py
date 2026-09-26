#!/usr/bin/env python3
"""R2 preparation: crops, reference sheets and the facts registered before any VLM run.

HA4M has two camera setups (a lab with parts in clear boxes, and a white room with
parts loose on the table). Setup is read from the image, not the subject id: the
white room's top-left wall is bright. One subject per setup is held out; its first
complete recording supplies the reference sheet, and none of its samples are scored.

Crops, all label-free and at native resolution:
  hands (Kinect)   the camera's own body tracking: a square of one shoulder width
                   around each wrist, union over both hands and both frames
  hands (YOLO)     the same rule on YOLOv8s-pose wrists and shoulders (RGB only)
  bench            per setup, the 5th-95th percentile box of all Kinect wrist
                   positions in that setup, expanded 1.2x -- a station ROI set once
A frame pair whose finder sees no wrist falls back to the full frame (counted).

    docker/run_harness.sh python3 experiments/r2_prep.py
"""
from __future__ import annotations

import collections
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("data/ha4m")
FW, FH = 2048, 1536
K_WRIST, K_SHOULDER = (7, 14), (5, 12)          # Azure Kinect body-tracking joints
Y_WRIST, Y_SHOULDER = (9, 10), (5, 6)            # COCO keypoints
MAX_PX = 451_584


def kinect(rec, f):
    p = ROOT / rec / "skel" / f"{f:06d}.txt"
    if not p.exists():
        return None
    j = {}
    for line in p.read_text().splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 12 and c[0] == "1":            # body 1 = the worker
            j[int(c[1])] = (float(c[10]), float(c[11]))
    if not j:
        return None
    return {"wrists": [j[k] for k in K_WRIST if k in j],
            "shoulders": [j[k] for k in K_SHOULDER if k in j]}


def yolo(pose, rec, f):
    p = pose.get(f"{rec}|{f}")
    if not p:
        return None
    kp = {int(k): v for k, v in p["kp"].items()}
    return {"wrists": [kp[k][:2] for k in Y_WRIST if kp[k][2] >= 0.3],
            "shoulders": [kp[k][:2] for k in Y_SHOULDER if kp[k][2] >= 0.3]}


def hand_box(parts):
    """Square of one shoulder width around each wrist, union over hands and frames."""
    sq = []
    for p in parts:
        if not p or not p["wrists"]:
            continue
        sw = (math.dist(*p["shoulders"]) if len(p["shoulders"]) == 2 else 0) or 160.0
        s = max(96.0, sw)
        sq += [(x - s / 2, y - s / 2, x + s / 2, y + s / 2) for x, y in p["wrists"]]
    if not sq:
        return None
    x1, y1 = max(0, min(b[0] for b in sq)), max(0, min(b[1] for b in sq))
    x2, y2 = min(FW, max(b[2] for b in sq)), min(FH, max(b[3] for b in sq))
    return [int(x1), int(y1), int(x2), int(y2)] if x2 - x1 > 32 and y2 - y1 > 32 else None


def setup_of(rec, f):
    from PIL import Image
    im = Image.open(ROOT / rec / "color" / f"{f:06d}.png").convert("L").crop((0, 0, 400, 300))
    return 2 if st.mean(im.getdata()) > 170 else 1


def main() -> None:
    from PIL import Image, ImageDraw, ImageFont
    man = [s for s in json.loads((ROOT / "manifest.json").read_text()) if s["complete"]]
    pose = json.loads((ROOT / "pose.json").read_text())
    setup = {}
    for s in man:
        if s["subject"] not in setup:
            setup[s["subject"]] = setup_of(s["rec"], s["frames"][0])
    by_setup = collections.defaultdict(list)
    for sub in sorted(setup):
        by_setup[setup[sub]].append(sub)
    held = {k: v[0] for k, v in by_setup.items()}           # first subject of each setup

    # the station ROI per setup, from every Kinect wrist in that setup (no labels)
    bench = {}
    for k in by_setup:
        xs, ys = [], []
        for s in man:
            if setup[s["subject"]] != k:
                continue
            for f in s["frames"]:
                kj = kinect(s["rec"], f)
                for x, y in (kj or {}).get("wrists", []):
                    xs.append(x); ys.append(y)
        xs.sort(); ys.sort()
        q = lambda v, p: v[int(p * (len(v) - 1))]
        x1, x2, y1, y2 = q(xs, .05), q(xs, .95), q(ys, .05), q(ys, .95)
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.2, (y2 - y1) * 1.2
        bench[k] = [int(max(0, cx - w / 2)), int(max(0, cy - h / 2)),
                    int(min(FW, cx + w / 2)), int(min(FH, cy + h / 2))]

    samples, fallback = [], collections.Counter()
    for s in man:
        k = setup[s["subject"]]
        kp = [kinect(s["rec"], f) for f in s["frames"]]
        yp = [yolo(pose, s["rec"], f) for f in s["frames"]]
        hk, hy = hand_box(kp), hand_box(yp)
        fallback["kinect"] += hk is None
        fallback["yolo"] += hy is None
        samples.append({**s, "setup": k, "held_out": s["subject"] == held[k],
                        "crop_kinect": hk, "crop_yolo": hy, "crop_bench": bench[k]})

    # reference sheets: the held-out subject's first complete recording, steps 1-12
    for k, sub in held.items():
        rec = min(s["rec"] for s in samples if s["subject"] == sub)
        cells = []
        for step in range(1, 13):
            s = next((x for x in samples if x["rec"] == rec and x["step"] == step), None)
            if s is None or s["crop_kinect"] is None:
                raise SystemExit(f"reference recording {rec} lacks a Kinect crop for step {step}")
            im = Image.open(ROOT / rec / "color" / f"{s['frames'][0]:06d}.png").convert("RGB")
            c = im.crop(tuple(s["crop_kinect"])).resize((300, 300), Image.BICUBIC)
            d = ImageDraw.Draw(c)
            try:
                font = ImageFont.load_default(size=40)
            except TypeError:
                font = ImageFont.load_default()
            d.rectangle((0, 0, 62, 50), fill=(255, 255, 255))
            d.text((8, 2), str(step), fill=(200, 0, 0), font=font)
            cells.append(c)
        sheet = Image.new("RGB", (4 * 300 + 3 * 6, 3 * 300 + 2 * 6), (255, 255, 255))
        for i, c in enumerate(cells):
            sheet.paste(c, ((i % 4) * 306, (i // 4) * 306))
        sheet.save(ROOT / f"refsheet_setup{k}.png")

    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1]) if b else None

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / (area(a) + area(b) - ix * iy)
    ev = [s for s in samples if not s["held_out"]]
    full_scale = math.sqrt(MAX_PX / (FW * FH))
    mag = lambda b: min(1.0, math.sqrt(MAX_PX / area(b))) / full_scale
    facts = {
        "samples_scored": len(ev), "held_out_subjects": held, "subjects_per_setup":
            {k: len(v) for k, v in by_setup.items()},
        "scored_per_setup": dict(collections.Counter(s["setup"] for s in ev)),
        "steps": dict(sorted(collections.Counter(s["step"] for s in ev).items())),
        "fallback_to_full_frame": dict(fallback),
        "bench_roi": bench,
        "crop_area_frac_median": {n: st.median(area(s[c]) / (FW * FH) for s in ev if s[c])
                                  for n, c in (("kinect", "crop_kinect"), ("yolo", "crop_yolo"),
                                               ("bench", "crop_bench"))},
        "magnification_vs_full_median": {n: st.median(mag(s[c]) for s in ev if s[c])
                                         for n, c in (("kinect", "crop_kinect"), ("yolo", "crop_yolo"),
                                                      ("bench", "crop_bench"))},
        "kinect_yolo_iou_median": st.median(iou(s["crop_kinect"], s["crop_yolo"]) for s in ev
                                            if s["crop_kinect"] and s["crop_yolo"]),
    }
    (ROOT / "r2_samples.json").write_text(json.dumps(samples))
    Path("results/r2").mkdir(parents=True, exist_ok=True)
    Path("results/r2/facts.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
