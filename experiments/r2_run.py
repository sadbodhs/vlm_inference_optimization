#!/usr/bin/env python3
"""R2 (RECIPE.md): hand crops on a manufacturing assembly (HA4M).

Reads data/ha4m/r2_samples.json (experiments/r2_prep.py). Every request carries the
setup's reference sheet first unless the arm is "names only" (-n).

  F / F-n    the full frame
  B          the fixed bench ROI of the setup
  HK / HK-n  the Kinect hand crop
  HY         the YOLOv8s-pose hand crop
  HKC        the Kinect hand crop + the full frame at <= 112,896 px

    docker/run_harness.sh python3 experiments/r2_run.py --arm arms/E_q3vl_8b.yaml
"""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

from _common import base_parser, resolve_arm
from r1_actor_crops import CONTEXT_PX, MAX_PX, png
from r1b_run import encode

ROOT = Path("data/ha4m")
ALL = ["F", "B", "HK", "HY", "HKC", "F-n", "HK-n"]
STEPS = ["pick up and place the carrier", "pick up and place the gear bearings (three)",
         "pick up and place the planet gears (three)", "pick up and place the carrier shaft",
         "pick up and place the sun shaft", "pick up and place the sun gear",
         "pick up and place the sun gear bearing", "pick up and place the ring bearing",
         "pick up block 2 and place it on block 1", "pick up and place the cover",
         "pick up and place the screws (two)",
         "pick up the Allen key, turn the screws, return the Allen key and the assembled gear train"]
STEP_LIST = "\n".join(f"{i}. {s[0].upper()}{s[1:]}" for i, s in enumerate(STEPS, 1))
ANSWER = "\nAnswer with the step number only, or 0 if the worker is doing none of these steps."
VIEW = {"F": "frames from the workstation camera",
        "B": "crops of the workbench area from the workstation camera",
        "HK": "close-up crops of the worker's hands from the workstation camera",
        "HY": "close-up crops of the worker's hands from the workstation camera",
        "HKC": "close-up crops of the worker's hands from the workstation camera"}


def question(arm: str) -> str:
    base = arm.split("-")[0]
    ctx = (" The last image is the whole camera view at the same moment, for context."
           if base == "HKC" else "")
    if arm.endswith("-n"):
        return (f"These two images are {VIEW[base]}, taken one second apart. A worker is "
                f"assembling a planetary (epicyclic) gear train.{ctx}\nWhich step is the "
                f"worker performing now?\nSteps:\n{STEP_LIST}{ANSWER}")
    return ("The first image is a reference sheet from this workstation: 12 numbered pictures, "
            "one for each step of a planetary (epicyclic) gear-train assembly, each showing a "
            f"worker performing that step. The next two images are {VIEW[base]}, taken one "
            f"second apart.{ctx}\nWhich step is the worker performing now?\nSteps:\n"
            f"{STEP_LIST}{ANSWER}")


def images(arm: str, s: dict):
    from PIL import Image
    base = arm.split("-")[0]
    paths = [ROOT / s["rec"] / "color" / f"{f:06d}.png" for f in s["frames"]]
    out = [] if arm.endswith("-n") else [((ROOT / f"refsheet_setup{s['setup']}.png").read_bytes(), MAX_PX)]
    box = {"B": s["crop_bench"], "HK": s["crop_kinect"], "HY": s["crop_yolo"],
           "HKC": s["crop_kinect"]}.get(base)
    for p in paths:
        if box is None:                        # F, or a finder that saw no wrist
            out.append((p.read_bytes(), MAX_PX))
        else:
            im = Image.open(p).convert("RGB")
            out.append((png(im.crop(tuple(box))), MAX_PX))
    if base == "HKC":
        out.append((paths[1].read_bytes(), CONTEXT_PX))
    return out


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--arms", default=",".join(ALL))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--r2-out", default="results/r2")
    args = p.parse_args()
    from bench.data import Sample
    from bench.harness import run_arm
    arm = resolve_arm(args)
    arm.max_tokens = 8
    arm.max_pixels = None
    samples = [s for s in json.loads((ROOT / "r2_samples.json").read_text()) if not s["held_out"]]
    if args.limit:
        samples = samples[::max(1, len(samples) // args.limit)][:args.limit]
    loop = asyncio.get_running_loop()
    for name in args.arms.split(","):
        def build(name=name):
            out = []
            for s in samples:
                frames, px = encode(images(name, s))
                out.append(Sample(id=f"{s['rec']}|{s['step']}|{s['frames'][0]}", question=question(name),
                                  answers=[str(s["step"])], frames_b64=frames,
                                  image_mime="jpeg", image_px=px))
            return out
        built = await loop.run_in_executor(None, build)
        summary = await run_arm(arm, built, mode="closed", concurrency=args.concurrency, scorer=None,
                                results_root=args.r2_out, run_id=f"{name}-{arm.id}",
                                unmeasured=["latency under load (closed loop)"])
        print(f"  {name:5s} n={len(built):5d}  prompt tokens mean "
              f"{summary['prompt_tokens']['mean']:.0f}  ok {summary['n_ok']}/{len(built)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
