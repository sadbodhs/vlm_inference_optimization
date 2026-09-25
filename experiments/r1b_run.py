#!/usr/bin/env python3
"""R1b (RECIPE.md): label-free group crops on gate-fired windows.

Reads results/r1b/windows.json (experiments/r1b_windows.py): the windows E7c's
track-motion gate fires and the proximity-group crops the tracker alone chose.

  F    one request per window: the full frame, E7's prompt
  M    one request per window: the full frame with every group's crop box in red
  GS   one request per group: the group's 2x crop
  GSC  one request per group: the 2x crop + the full frame at <= 112,896 px
  GO   one request per window: every group's crop (two frames each) + one low-res full frame

A fired window with no group falls back to the full frame in every arm.

    docker/run_harness.sh python3 experiments/r1b_run.py --arm arms/E_q3vl_4b.yaml
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

from _common import base_parser, resolve_arm
from e7_vlm import QUESTION
from r1_actor_crops import CONTEXT_PX, MAX_PX, OPTIONS, Q_CROP, Q_CTX, png

ALL = ["F", "M", "GS", "GSC", "GO"]
Q_MARKS = ("These two frames were taken one second apart by a fixed security camera. Red "
           "boxes mark the people, with any vehicle next to them.\n"
           "What are the people inside the red boxes doing?\n" + OPTIONS)


def q_go(k: int) -> str:
    parts = ", ".join(f"images {2*i+1}-{2*i+2} show group {i+1}" for i in range(k))
    return ("These images come from a fixed security camera. The first "
            f"{2*k} are close-up crops of {k} group{'s' if k > 1 else ''} of people, two per "
            f"group taken one second apart: {parts}. The last image is the whole camera view "
            "at the same moment, for context.\nWhich of the following are happening in any of "
            "the close-ups?\n" + OPTIONS)


def encode(images):
    from bench.imaging import resize_to_budget
    frames, px = [], 0
    for raw, cap in images:
        data, _, h, w = resize_to_budget(raw, cap)
        frames.append(base64.b64encode(data).decode())
        px += h * w
    return frames, px


def samples_for(arm: str, win: dict, frame_dir: str):
    """[(sample id, question, [(bytes, cap)])] for one fired window under one arm."""
    from PIL import Image, ImageDraw
    raws = [(Path(frame_dir) / win["clip"] / f"{f:05d}.jpg").read_bytes() for f in win["frames"]]
    wid = win["id"]
    full = (f"{wid}|full", QUESTION, [(r, MAX_PX) for r in raws])
    if arm == "F" or not win["crops"]:
        return [full]
    ims = [Image.open(io.BytesIO(r)).convert("RGB") for r in raws]
    if arm == "M":
        out = []
        for im in ims:
            d = ImageDraw.Draw(im)
            for c in win["crops"]:
                d.rectangle(c, outline=(255, 0, 0), width=5)
            out.append((png(im), MAX_PX))
        return [(f"{wid}|marked", Q_MARKS, out)]
    crops = [[(png(im.crop(tuple(c))), MAX_PX) for im in ims] for c in win["crops"]]
    ctx = (raws[1], CONTEXT_PX)
    if arm == "GS":
        return [(f"{wid}|g{i}", Q_CROP, cr) for i, cr in enumerate(crops)]
    if arm == "GSC":
        return [(f"{wid}|g{i}", Q_CTX, cr + [ctx]) for i, cr in enumerate(crops)]
    if arm == "GO":
        return [(f"{wid}|all", q_go(len(crops)), [x for cr in crops for x in cr] + [ctx])]
    raise ValueError(arm)


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--windows", default="results/r1b/windows.json")
    p.add_argument("--frame-dir", default="data/meva/frames")
    p.add_argument("--arms", default=",".join(ALL))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="windows, for a smoke run")
    p.add_argument("--r1b-out", default="results/r1b")
    args = p.parse_args()

    from bench.data import Sample
    from bench.harness import run_arm
    arm = resolve_arm(args)
    arm.max_tokens = 16
    arm.max_pixels = None
    wins = json.loads(Path(args.windows).read_text())["windows"]
    if args.limit:
        wins = wins[:args.limit]
    loop = asyncio.get_running_loop()
    for name in args.arms.split(","):
        def build(name=name):
            out = []
            for w in wins:
                for sid, q, imgs in samples_for(name, w, args.frame_dir):
                    frames, px = encode(imgs)
                    out.append(Sample(id=sid, question=q, answers=w["truth"] or ["N"],
                                      frames_b64=frames, image_mime="jpeg", image_px=px))
            return out
        samples = await loop.run_in_executor(None, build)
        summary = await run_arm(
            arm, samples, mode="closed", concurrency=args.concurrency, scorer=None,
            results_root=args.r1b_out, run_id=f"{name}-{arm.id}",
            unmeasured=["latency under load (closed loop, not a capacity measurement)"])
        print(f"  {name:4s} requests {len(samples):5d}  prompt tokens mean "
              f"{summary['prompt_tokens']['mean']:.0f}  ok {summary['n_ok']}/{len(samples)}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
