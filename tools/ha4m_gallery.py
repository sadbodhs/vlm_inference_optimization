#!/usr/bin/env python3
"""A viewing gallery of HA4M: for a few steps in each camera setup, the full frame
beside the wide and tight hand crops R2b compares (from the Kinect's body tracking).
For looking at, not for measurement.

    docker/run_harness.sh python3 tools/ha4m_gallery.py IDU001V001 IDU023V001
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import r2_prep as P                                     # noqa: E402

STEPS = (3, 6, 10)                                      # planet gears, sun gear, cover
NAMES = {3: "3 planet gears", 6: "6 sun gear", 10: "10 cover"}


def box(kp, width):
    sq = []
    for p in kp:
        if not p or not p["wrists"]:
            continue
        sw = (math.dist(*p["shoulders"]) if len(p["shoulders"]) == 2 else 0) or 160.0
        s = max(96.0, sw) * width
        sq += [(x - s / 2, y - s / 2, x + s / 2, y + s / 2) for x, y in p["wrists"]]
    return [int(max(0, min(b[0] for b in sq))), int(max(0, min(b[1] for b in sq))),
            int(min(P.FW, max(b[2] for b in sq))), int(min(P.FH, max(b[3] for b in sq)))]


def main() -> None:
    from PIL import Image, ImageDraw, ImageFont
    recs = sys.argv[1:]
    H = 300
    rows = []
    for rec in recs:
        d = P.ROOT / rec
        lab = {int(a): int(b) for a, b, *_ in (l.split() for l in (d / "labels.txt").read_text().splitlines() if l.strip())}
        frames = sorted(int(p.stem) for p in (d / "color").glob("*.png"))
        for step in STEPS:
            f = next(x for x in frames if lab.get(x) == step)
            im = Image.open(d / "color" / f"{f:06d}.png").convert("RGB")
            kp = [P.kinect(rec, f)]
            tiles = [("full frame", im),
                     ("wide hands (3x)", im.crop(tuple(box(kp, 3.0)))),
                     ("tight hands (1x)", im.crop(tuple(box(kp, 1.0))))]
            row = []
            for title, t in tiles:
                t = t.resize((max(1, round(t.width * H / t.height)), H))
                canvas = Image.new("RGB", (t.width, H + 36), (255, 255, 255))
                canvas.paste(t, (0, 36))
                ImageDraw.Draw(canvas).text((6, 4), f"{rec} · step {NAMES[step]} · {title}", fill=(20, 20, 20),
                                            font=ImageFont.load_default(size=18))
                row.append(canvas)
            rows.append(row)
    W = max(sum(c.width for c in r) + 12 * (len(r) - 1) for r in rows)
    out = Image.new("RGB", (W, len(rows) * (H + 36 + 14)), (255, 255, 255))
    y = 0
    for r in rows:
        x = 0
        for c in r:
            out.paste(c, (x, y)); x += c.width + 12
        y += H + 36 + 14
    out.save(P.ROOT / "gallery.jpg", quality=85)
    print(f"wrote {P.ROOT/'gallery.jpg'} {out.size}")


if __name__ == "__main__":
    main()
