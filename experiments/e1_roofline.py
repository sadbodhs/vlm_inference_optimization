#!/usr/bin/env python3
"""E1 -- single-stream decode vs the memory-bandwidth roofline.

Question: does this arm's decode rate land where the RTX 3090's 936 GB/s says it
must, and if not, how much is left on the table?

Decode reads every weight once per token, so the ceiling is
    tok/s <= HBM_BW / weight_bytes
This is the first experiment for a reason: a measured rate *above* the roofline is
physically impossible, so it is a harness bug -- usually a counted warmup or a
mis-attributed first token. E1 is as much a test of the harness as of the server.

    python3 experiments/e1_roofline.py --arm arms/B_vllm_awq.yaml --dry-run
"""
from __future__ import annotations

import asyncio

from _common import base_parser, resolve_arm, table, write_sweep


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--n", type=int, default=12, help="requests (concurrency 1)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    from bench.data import synthetic
    from bench.harness import run_arm
    from bench.metrics import HBM_BW_GB_S, decode_roofline_tok_s

    arm = resolve_arm(args)
    arm.max_tokens = args.max_tokens
    tag = args.tag or f"e1-roofline-{arm.id}"

    # Small image: E1 is about decode, so vision cost is deliberately minimised.
    samples = synthetic(max(args.n, 1), width=224, height=224)

    if args.warmup:
        await run_arm(arm, samples[: args.warmup], mode="sequential",
                      results_root=args.out, run_id=f"{tag}-warmup")

    s = await run_arm(
        arm, samples, mode="sequential", results_root=args.out, run_id=tag,
        unmeasured=["batched decode", "accuracy", "long-context decode"],
    )

    measured = None
    if s["tpot_ms"]["p50"]:
        measured = 1000.0 / s["tpot_ms"]["p50"]

    roof = decode_roofline_tok_s(arm.roofline_bytes) if arm.roofline_bytes else None
    row = {
        "arm": arm.id,
        "weight_GB": (arm.weight_bytes / 1e9) if arm.weight_bytes else None,
        "decode_weight_GB": (arm.roofline_bytes / 1e9) if arm.roofline_bytes else None,
        "tpot_p50_ms": s["tpot_ms"]["p50"],
        "decode_tok_s": measured,
        "roofline_tok_s": roof,
        "pct_of_roofline": (100.0 * measured / roof) if (roof and measured) else None,
        "ttft_p50_ms": s["ttft_ms"]["p50"],
    }

    print(f"\nE1 · single-stream decode vs roofline   (HBM {HBM_BW_GB_S:.0f} GB/s)")
    table([row], [("arm", "arm"), ("weight_GB", "weights GB"),
                  ("decode_tok_s", "tok/s"), ("roofline_tok_s", "roofline"),
                  ("pct_of_roofline", "% of roof"), ("ttft_p50_ms", "TTFT p50")])

    if row["pct_of_roofline"] and row["pct_of_roofline"] > 100:
        print("\n  !! ABOVE ROOFLINE -- this is impossible on real hardware.")
        print("     Either weight_bytes in the arm YAML is wrong, or the harness is")
        print("     counting a token it should not. Fix before trusting any other run.")

    write_sweep(args.out, tag, [row],
                {"experiment": "e1-roofline", "hbm_gb_s": HBM_BW_GB_S,
                 "arm": arm.id, "n": args.n, "max_tokens": args.max_tokens})


if __name__ == "__main__":
    asyncio.run(main())
