#!/usr/bin/env python3
"""Concept figure for Track B: one MEVA moment, three ways of choosing the crop.

  E7    one box around every detected person (the ROI arm of E7-E7d)
  R1    the activity's annotated participants, 2x (an ideal crop, needs labels)
  R1b   proximity groups of tracked people, 2x (label-free, what a pipeline can do)

The sample is picked, not hand-chosen: an R1 positive involving two or more people,
in a window R1b's gate fired, with at least two tracker groups -- the case where
the three strategies differ most.

    docker/run_harness.sh python3 tools/concept_crops.py      # -> docs/img/crop-strategies.jpg
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from e7_vlm import DET_SIZE, VLM_OFFSETS, WINDOW, load_dets, roi_box   # noqa: E402
from r1_actor_crops import expand                                     # noqa: E402

COL = {"E7": (230, 159, 0), "R1": (42, 100, 180), "R1b": (31, 138, 82)}


def main() -> None:
    from PIL import Image, ImageDraw, ImageFont
    r1 = json.loads(Path("results/r1/samples.json").read_text())["samples"]
    wins = {w["id"]: w for w in json.loads(Path("results/r1b/windows.json").read_text())["windows"]}
    cands = [s for s in r1 if s["kind"] == "pos" and s["cls"] == "2+ people"
             and 70 <= (s["actor_h"] or 0) <= 160 and f"{s['clip']}|{s['w']}" in wins
             and len(wins[f"{s['clip']}|{s['w']}"]["crops"]) >= 2]
    if not cands:
        raise SystemExit("no sample matches")
    s = sorted(cands, key=lambda x: (-len(wins[f"{x['clip']}|{x['w']}"]["crops"]), x["id"]))[0]
    w = wins[f"{s['clip']}|{s['w']}"]
    f = s["frames"][1]
    im = Image.open(Path("data/meva/frames") / s["clip"] / f"{f:05d}.jpg").convert("RGB")
    dets = load_dets(Path("data/meva/det") / f"{s['clip']}.{DET_SIZE}.jsonl.gz")
    e7 = roi_box(dets.get(s["frames"][0], []), dets.get(s["frames"][1], []), im.width, im.height)
    r1box = expand(s["box"], 2.0)
    groups = [tuple(c) for c in w["crops"]]

    over = im.copy()
    d = ImageDraw.Draw(over)
    font = ImageFont.load_default(size=30)
    if e7:
        d.rectangle(e7, outline=COL["E7"], width=8)
    d.rectangle(r1box, outline=COL["R1"], width=8)
    for g in groups:
        d.rectangle(g, outline=COL["R1b"], width=6)
    H = 420
    main_im = over.resize((round(over.width * 760 / over.height), 760))
    tiles = [("E7: one box around everyone", im.crop(e7) if e7 else im, COL["E7"]),
             ("R1: the labelled participants, 2x", im.crop(r1box), COL["R1"])]
    tiles += [(f"R1b: tracker group {i+1}, no labels", im.crop(g), COL["R1b"]) for i, g in enumerate(groups[:3])]
    small = []
    for title, t, c in tiles:
        t = t.resize((max(1, round(t.width * (H // 2 - 40) / t.height)), H // 2 - 40))
        canvas = Image.new("RGB", (max(t.width, 330), H // 2), (255, 255, 255))
        canvas.paste(t, (0, 36))
        dd = ImageDraw.Draw(canvas)
        dd.rectangle((0, 0, 14, 28), fill=c)
        dd.text((20, 2), title, fill=(25, 25, 25), font=ImageFont.load_default(size=19))
        small.append(canvas)
    cols = 2
    rows = [small[i:i + cols] for i in range(0, len(small), cols)]
    right_w = max(sum(c.width for c in r) + 10 for r in rows)
    out = Image.new("RGB", (main_im.width + 20 + right_w, max(760, len(rows) * (H // 2 + 10))), (255, 255, 255))
    out.paste(main_im, (0, 0))
    y = 0
    for r in rows:
        x = main_im.width + 20
        for c in r:
            out.paste(c, (x, y)); x += c.width + 10
        y += H // 2 + 10
    Path("docs/img").mkdir(parents=True, exist_ok=True)
    out.save("docs/img/crop-strategies.jpg", quality=84)
    print(f"sample {s['id']} ({s['type']}), {len(groups)} groups -> docs/img/crop-strategies.jpg {out.size}")


if __name__ == "__main__":
    main()
