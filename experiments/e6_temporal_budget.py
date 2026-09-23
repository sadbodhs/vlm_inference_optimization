#!/usr/bin/env python3
"""E6 -- the temporal budget frontier: how many frames do you actually need?

E3 asked how many vision tokens one image needs. A clip costs
**frames x tokens-per-frame**, and a fixed context budget makes those two compete:
at ~1,000 tokens per frame, 8,192 tokens of context buys 8 frames and no more.

So the deployment question is not "how many frames" but how to spend a budget:
is 2 frames at 1,000 tokens better than 8 frames at 250?

TempCompass makes the answer task-dependent by construction. Its dimensions split
into questions a single frame can plausibly answer (`action` -- largely
appearance) and ones it cannot (`direction`, `speed`, `order`, `attribute_change`
-- all require motion or comparison). Same clips, same format, same scorer; only
the question type varies.

    python3 experiments/e6_temporal_budget.py --arm arms/B0_vllm_awq_clean.yaml \\
        --manifest data/tempcompass/manifest.jsonl --frames 1,2,4,8

Chance level is ~0.33 (mostly 3-option), so read scores against that floor.
"""
from __future__ import annotations

import asyncio
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from _common import base_parser, resolve_arm, table, write_sweep


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--frames", default="1,2,4,8",
                   help="frame counts per clip to sweep")
    p.add_argument("--max-pixels-per-frame", type=int, default=200704,
                   help="token budget PER FRAME; total scales with frame count")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--prompt-suffix",
                   default="\nAnswer with the option letter only.")
    args = p.parse_args()

    from bench.data import Sample
    from bench.frames import FrameError, sample_frames
    from bench.harness import run_arm

    arm = resolve_arm(args)
    arm.max_tokens = args.max_tokens
    arm.max_pixels = None            # per-frame budget is applied client-side
    tag = args.tag or f"e6-temporal-{arm.id}"
    counts = [int(x) for x in args.frames.split(",")]

    root = Path(args.manifest).parent
    recs = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    recs = recs[: args.limit]
    dims = sorted({r.get("dim") for r in recs if r.get("dim")})
    print(f"{len(recs)} clips  dimensions={dims}")
    print(f"per-frame budget {args.max_pixels_per_frame} px\n")

    rows = []
    for n in counts:
        # Extraction and resizing happen here, before any request is timed.
        batch, skipped = [], 0
        for r in recs:
            try:
                frames = sample_frames(root / r["video"], n)
            except FrameError:
                skipped += 1
                continue
            s = Sample(id=r["id"], question=r["question"], answers=r["answers"])
            s = s.with_frames(frames, args.max_pixels_per_frame)
            batch.append(s)
        if not batch:
            print(f"  frames={n}: no clips decoded, skipping")
            continue

        summary = await run_arm(
            arm, batch, mode="sequential", scorer="mc",
            prompt_suffix=args.prompt_suffix,
            results_root=args.out, run_id=f"{tag}-f{n}",
            unmeasured=["throughput under load", "audio", "frame-sampling policy"],
        )

        # Per-dimension accuracy: the whole point is that the curves differ.
        by_id = {r["id"]: r for r in recs}
        outs = [json.loads(l) for l in
                (Path(summary["out_dir"]) / "outputs.jsonl").read_text().splitlines() if l.strip()]
        from bench.scorers import SCORERS
        mc = SCORERS["mc"]
        per_dim = defaultdict(list)
        for o in outs:
            rec = by_id.get(o["sample_id"])
            if rec and o["ok"]:
                per_dim[rec.get("dim")].append(mc(o["text"], rec["answers"]))

        row = {
            "frames": n,
            "prompt_tokens": summary["prompt_tokens"]["mean"],
            "ttft_p50_ms": summary["ttft_ms"]["p50"],
            "accuracy": summary["accuracy"]["score"],
            "n_scored": summary["accuracy"].get("n_scored"),
            "skipped_clips": skipped,
        }
        for d in dims:
            row[d] = st.mean(per_dim[d]) if per_dim.get(d) else None
        rows.append(row)
        print(f"  frames={n:<3} tok={row['prompt_tokens']:>7.0f}  "
              f"TTFT {row['ttft_p50_ms']:>7.1f} ms  acc={row['accuracy']:.3f}")

    print(f"\nE6 · temporal budget   per-frame budget {args.max_pixels_per_frame} px")
    cols = [("frames", "frames"), ("prompt_tokens", "prompt tok"),
            ("ttft_p50_ms", "TTFT p50"), ("accuracy", "overall")]
    cols += [(d, d[:12]) for d in dims]
    table(rows, cols)

    if len(rows) >= 2:
        base, last = rows[0], rows[-1]
        print(f"\n  {base['frames']} -> {last['frames']} frames: "
              f"TTFT {base['ttft_p50_ms']:.0f} -> {last['ttft_p50_ms']:.0f} ms "
              f"({last['ttft_p50_ms']/max(base['ttft_p50_ms'],1e-9):.1f}x), "
              f"accuracy {base['accuracy']:.3f} -> {last['accuracy']:.3f}")
        print("\n  per-dimension gain from more frames (chance ~0.33):")
        for d in dims:
            if base.get(d) is not None and last.get(d) is not None:
                print(f"    {d:18} {base[d]:.3f} -> {last[d]:.3f}   {last[d]-base[d]:+.3f}")

    write_sweep(args.out, tag, rows,
                {"experiment": "e6-temporal-budget", "arm": arm.id,
                 "manifest": args.manifest, "frames": counts,
                 "max_pixels_per_frame": args.max_pixels_per_frame,
                 "limit": args.limit, "dims": dims, "chance_level": 1/3,
                 "prompt_suffix": args.prompt_suffix})


if __name__ == "__main__":
    asyncio.run(main())
