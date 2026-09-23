#!/usr/bin/env python3
"""E7 stage 4: how many cascade calls per second can this server take, fresh?

"Cameras per 3090" is capacity divided by demand. Demand comes from the gates
(e7_gate.py); capacity has to come from THIS input -- two frames per request, at
the full-frame or ROI size -- not from E4's single-frame number, because prefill
cost scales with vision tokens and a two-frame window carries ~3.5x E4's tokens.

Open loop (Poisson), a disjoint slice of windows per rate so nothing is re-sent.
Usable capacity is the highest offered rate whose verdict is not `saturated` and
whose TTFT p99 is inside the 2 s freshness budget E4 used -- a surveillance answer
about a frame older than that is not an answer about the present.

    docker/run_harness.sh python3 experiments/e7_capacity.py --arm arms/B0_vllm_awq_clean.yaml \\
        --which full --rates 1,2,3,4,6
"""
from __future__ import annotations

import asyncio
import io
import json
import random
from pathlib import Path

from _common import base_parser, resolve_arm, table, write_sweep
from e7_vlm import (DET_SIZE, QUESTION, VLM_OFFSETS, WINDOW, crop_jpeg, load_dets,
                    roi_box)

FRESH_MS = 2000.0


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--selection", default="results/e7/selection.json")
    p.add_argument("--index", default="data/meva/index.json")
    p.add_argument("--frame-dir", default="data/meva/frames")
    p.add_argument("--det-dir", default="data/meva/det")
    p.add_argument("--which", choices=["full", "roi"], default="full")
    p.add_argument("--rates", default="1,2,3,4,6")
    p.add_argument("--n", type=int, default=80, help="requests per rate")
    p.add_argument("--max-pixels", type=int, default=451584)
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    from bench.data import Sample
    from bench.harness import run_arm
    from bench.metrics import SLO
    from PIL import Image

    arm = resolve_arm(args)
    arm.max_tokens = args.max_tokens
    arm.max_pixels = None
    rates = [float(r) for r in args.rates.split(",")]
    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}

    # Windows a cascade would actually send: ones with something detected. Sampled
    # across all clips so no rate is one camera's content.
    pool = []
    for c in sel:
        dets = load_dets(Path(args.det_dir) / f"{c['clip']}.{DET_SIZE}.jsonl.gz")
        for w in range(idx[c["clip"]]["n_frames"] // WINDOW):
            fr = [w * WINDOW + o for o in VLM_OFFSETS]
            if dets.get(fr[0]) or dets.get(fr[1]):
                pool.append((c["clip"], w, fr, dets.get(fr[0], []), dets.get(fr[1], [])))
    random.Random(0).shuffle(pool)
    need = args.n * len(rates)
    if len(pool) < need:
        raise SystemExit(f"only {len(pool)} windows with detections; need {need} for zero reuse")

    samples = []
    for clip, w, fr, da, db in pool[:need]:
        raws = [(Path(args.frame_dir) / clip / f"{f:05d}.jpg").read_bytes() for f in fr]
        if args.which == "roi":
            W, H = Image.open(io.BytesIO(raws[0])).size
            raws = [crop_jpeg(r, roi_box(da, db, W, H)) for r in raws]
        s = Sample(id=f"{clip}|{w}", question=QUESTION, answers=["N"])
        samples.append(s.with_frames(raws, args.max_pixels))

    tag = args.tag or f"e7-capacity-{args.which}-{arm.id}"
    rows = []
    for i, rate in enumerate(rates):
        sl = samples[i * args.n:(i + 1) * args.n]
        s = await run_arm(arm, sl, mode="open", rate_qps=rate, slo=SLO(ttft_ms=FRESH_MS),
                          results_root=args.out, run_id=f"{tag}-qps{rate:g}",
                          unmeasured=["accuracy (see e7_vlm.py)", "detector sharing the GPU"])
        h = s["harness"]
        rows.append({"offered_qps": rate, "achieved_qps": h["achieved_qps"],
                     "prompt_tokens": s["prompt_tokens"]["mean"],
                     "ttft_p50_ms": s["ttft_ms"]["p50"], "ttft_p99_ms": s["ttft_ms"]["p99"],
                     "goodput_req_s": s["goodput_req_s"], "verdict": h["verdict"]})
        print(f"  qps={rate:<5g} {h['verdict']:<14} TTFT p99 {s['ttft_ms']['p99']:.0f} ms")

    print(f"\nE7 capacity · {args.which} · {rows[0]['prompt_tokens']:.0f} prompt tokens/request")
    table(rows, [("offered_qps", "offered"), ("achieved_qps", "achieved"),
                 ("ttft_p50_ms", "TTFT p50"), ("ttft_p99_ms", "TTFT p99"),
                 ("goodput_req_s", "goodput"), ("verdict", "verdict")])
    ok = [r for r in rows if r["verdict"] == "not-saturated" and r["ttft_p99_ms"] <= FRESH_MS]
    usable = max((r["achieved_qps"] for r in ok), default=None)
    print(f"\n  usable capacity (not saturated, TTFT p99 <= {FRESH_MS:.0f} ms): "
          f"{usable if usable is None else f'{usable:.2f} windows/s'}")
    write_sweep(args.out, tag, rows,
                {"experiment": "e7-capacity", "arm": arm.id, "which": args.which,
                 "rates": rates, "n_per_rate": args.n, "fresh_ms": FRESH_MS,
                 "usable_windows_per_s": usable, "max_pixels": args.max_pixels})


if __name__ == "__main__":
    asyncio.run(main())
