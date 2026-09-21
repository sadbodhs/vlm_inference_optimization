#!/usr/bin/env python3
"""E2 -- vision tokens vs TTFT: where the knee is, and who is actually spending it.

The token count of an image is a policy choice, not a property of the image, and it
is the largest lever on TTFT that no serving benchmark reports. E2 sweeps image
geometry and measures:

  vision_tokens   derived by subtraction -- no serving stack reports it directly.
                  A text-only control run gives text_tokens for the same prompt;
                  vision_tokens = prompt_tokens(image) - prompt_tokens(text-only).
  TTFT            at concurrency 1, so nothing is hiding inside queueing delay.

The output is the curve TTFT(vision_tokens) plus its linear fit. The intercept is
everything that does not scale with the image -- connection, tokenizer, sampler
setup, cold caches. The slope is the marginal cost of a vision token. Reporting both
is what separates "the encoder is slow" from "the encoder is fine and you are
sending 4000 tokens".

    python3 experiments/e2_ttft_vs_tokens.py --arm arms/B_vllm_awq.yaml --dry-run
"""
from __future__ import annotations

import asyncio

from _common import base_parser, resolve_arm, table, write_sweep


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--sizes", default="224,448,672,896,1120,1344",
                   help="square image edge lengths in pixels")
    p.add_argument("--n", type=int, default=8, help="requests per size")
    p.add_argument("--max-tokens", type=int, default=8,
                   help="kept tiny: E2 is about TTFT, not decode")
    args = p.parse_args()

    from bench.data import synthetic, text_only
    from bench.harness import run_arm

    arm = resolve_arm(args)
    arm.max_tokens = args.max_tokens
    tag = args.tag or f"e2-ttft-{arm.id}"
    sizes = [int(x) for x in args.sizes.split(",")]

    # --- control: same prompt, no image -------------------------------
    ctrl = await run_arm(arm, text_only(args.n), mode="sequential",
                         results_root=args.out, run_id=f"{tag}-textonly",
                         unmeasured=["anything involving an image"])
    text_tokens = ctrl["prompt_tokens"]["mean"]
    print(f"\ntext-only control: prompt_tokens={text_tokens}, "
          f"TTFT p50={ctrl['ttft_ms']['p50']:.1f} ms")

    rows = []
    for edge in sizes:
        s = await run_arm(
            arm, synthetic(args.n, width=edge, height=edge), mode="sequential",
            results_root=args.out, run_id=f"{tag}-{edge}px",
            unmeasured=["accuracy at this resolution", "batched TTFT"],
        )
        pt = s["prompt_tokens"]["mean"]
        vt = (pt - text_tokens) if (pt and text_tokens) else None
        rows.append({
            "edge_px": edge,
            "image_px": edge * edge,
            "prompt_tokens": pt,
            "vision_tokens": vt,
            "ttft_p50_ms": s["ttft_ms"]["p50"],
            "ttft_p95_ms": s["ttft_ms"]["p95"],
            "ms_per_vision_token": (s["ttft_ms"]["p50"] / vt) if vt else None,
        })
        print(f"  {edge:>5}px  vision_tokens={vt}  TTFT p50={s['ttft_ms']['p50']:.1f} ms")

    print("\nE2 · TTFT vs vision tokens (concurrency 1)")
    table(rows, [("edge_px", "edge px"), ("vision_tokens", "vis tok"),
                 ("prompt_tokens", "prompt tok"), ("ttft_p50_ms", "TTFT p50"),
                 ("ttft_p95_ms", "TTFT p95"), ("ms_per_vision_token", "ms/vis tok")])

    fit = None
    pts = [(r["vision_tokens"], r["ttft_p50_ms"]) for r in rows
           if r["vision_tokens"] and r["ttft_p50_ms"]]
    if len(pts) >= 2:
        import numpy as np

        x = np.array([a for a, _ in pts], dtype=float)
        y = np.array([b for _, b in pts], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        ss = 1 - (resid**2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
        fit = {"slope_ms_per_token": float(slope),
               "intercept_ms": float(intercept), "r2": float(ss)}
        print(f"\n  TTFT ≈ {intercept:.1f} ms + {slope*1000:.2f} ms per 1k vision tokens"
              f"   (R²={ss:.3f})")
        print(f"  fixed floor {intercept:.1f} ms is independent of the image -- that is")
        print("  the part no amount of token reduction will ever buy back.")

    write_sweep(args.out, tag, rows,
                {"experiment": "e2-ttft-vs-tokens", "arm": arm.id, "sizes": sizes,
                 "text_only_prompt_tokens": text_tokens,
                 "text_only_ttft_p50_ms": ctrl["ttft_ms"]["p50"], "fit": fit})


if __name__ == "__main__":
    asyncio.run(main())
