#!/usr/bin/env python3
"""E8 recognition pass: the window sent as images or as one video (PLAN.md 14).

E7's VLM stage, full frame, every window -- only the request changes:

  img2   frames 18, 48 as two images (E7-E7d)
  vid2   the same two frames as one video, fps 1. Qwen merges each pair of video
         frames into one set of tokens, so this is ~43% fewer prompt tokens
  vid8   frames 6, 12, ..., 48 as one video, fps 5

EVS (--video-pruning-rate) is a SERVER flag, so a pruned arm is vid8 against a
server started with it; --evs records the rate in the run name, it does not set it.

    docker/run_harness.sh python3 experiments/e8_vlm.py --arm arms/V_vllm_video.yaml \\
        --inputs img2,vid2,vid8
    docker/run_harness.sh python3 experiments/e8_vlm.py --arm arms/V_vllm_video.yaml \\
        --inputs vid8 --evs 0.75      # server started with --video-pruning-rate 0.75
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from _common import base_parser, resolve_arm
from e7_vlm import QUESTION, WINDOW, window_groups

# input -> (frame offsets inside the 60-frame window, video fps or None for images).
# Every input ends on frame 48, so a live run sends all of them at the same moment.
INPUTS = {
    "img2": ((18, 48), None),
    "vid2": ((18, 48), 1),
    "vid8": (tuple(range(6, 49, 6)), 5),
}
# The image wording ("two frames ... one second apart") would be false for a clip.
VIDEO_QUESTION = QUESTION.replace("These two frames were taken one second apart",
                                  "This short clip was taken")
assert VIDEO_QUESTION != QUESTION


def run_name(inp: str, evs: float, arm_id: str) -> str:
    return f"e8-{inp}{f'-e{round(100 * evs)}' if evs else ''}-{arm_id}"


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--selection", default="results/e7/selection.json")
    p.add_argument("--index", default="data/meva/index.json")
    p.add_argument("--frame-dir", default="data/meva/frames5")
    p.add_argument("--inputs", default="img2,vid2,vid8")
    p.add_argument("--evs", type=float, default=0.0, help="the server's --video-pruning-rate")
    p.add_argument("--max-pixels", type=int, default=451584, help="per-frame cap")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit-clips", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    from bench.data import Sample
    from bench.harness import run_arm

    arm = resolve_arm(args)
    arm.max_tokens = args.max_tokens
    arm.max_pixels = None          # per-frame cap applied client-side
    base_extra = dict(arm.extra_body)
    sel = json.load(open(args.selection))["clips"]
    if args.limit_clips:
        sel = sel[: args.limit_clips]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}

    for inp in args.inputs.split(","):
        offsets, fps = INPUTS[inp]
        video = fps is not None
        arm.extra_body = ({**base_extra, "media_io_kwargs": {"video": {"fps": fps}}}
                          if video else base_extra)
        samples = []
        for c in sel:
            clip, meta = c["clip"], idx[c["clip"]]
            for w in range(meta["n_frames"] // WINDOW):
                raws = [(Path(args.frame_dir) / clip / f"{w * WINDOW + o:05d}.jpg").read_bytes()
                        for o in offsets]
                gold = window_groups(meta["activities"], w)
                s = Sample(id=f"{clip}|{w}", question=VIDEO_QUESTION if video else QUESTION,
                           answers=gold or ["N"])
                samples.append(s.with_frames(raws, args.max_pixels, as_video=video))
        print(f"\n[{inp}{f' evs {args.evs}' if args.evs else ''}] {len(samples)} windows")

        summary = await run_arm(
            arm, samples, mode="closed", concurrency=args.concurrency, scorer=None,
            results_root=args.out, run_id=run_name(inp, args.evs, arm.id),
            unmeasured=["latency: closed loop at fixed concurrency is for answers, "
                        "not a capacity measurement (see e7c.py live)",
                        "batch-composition effects on greedy decoding"],
        )
        print(f"  prompt tokens mean {summary['prompt_tokens']['mean']:.0f}   "
              f"ok {summary['n_ok']}/{len(samples)}")
        if summary["n_ok"] < len(samples):
            raise SystemExit(f"{len(samples) - summary['n_ok']} requests failed")


if __name__ == "__main__":
    asyncio.run(main())
