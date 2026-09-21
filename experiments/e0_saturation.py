#!/usr/bin/env python3
"""E0 -- saturation discovery, and proof the harness is not the bottleneck.

R2 says saturate before you measure. That is easy to say and almost never checked,
so E0 runs first and produces the number every later throughput claim depends on:
the arrival rate at which this arm stops keeping up.

It sweeps offered QPS upward and watches three things at each rate:

  achieved/offered   < 1 means the server is falling behind  -> saturated
  loadgen lag drift  > 0 and growing means the *client* fell behind -> invalid run
  TTFT p99           the knee is where queueing delay starts dominating

A rate whose verdict is `client-bound` is discarded, not reported. That distinction
is the whole point: both failure modes flatten the throughput curve identically.

    python3 experiments/e0_saturation.py --arm arms/B_vllm_awq.yaml --dry-run
"""
from __future__ import annotations

import asyncio

from _common import base_parser, resolve_arm, table, write_sweep


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--rates", default="1,2,4,8,16,32", help="offered QPS, comma separated")
    p.add_argument("--n", type=int, default=60, help="requests per rate")
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--height", type=int, default=448)
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()

    from bench.data import synthetic
    from bench.harness import run_arm

    arm = resolve_arm(args)
    arm.max_tokens = args.max_tokens
    tag = args.tag or f"e0-saturation-{arm.id}"
    rates = [float(x) for x in args.rates.split(",")]
    samples = synthetic(min(args.n, 32), width=args.width, height=args.height)

    rows = []
    for rate in rates:
        s = await run_arm(
            arm, samples, mode="open", rate_qps=rate,
            repeats=max(args.n // len(samples), 1),
            results_root=args.out, run_id=f"{tag}-qps{rate:g}",
            unmeasured=["accuracy", "multi-turn reuse", "mixed request sizes"],
        )
        h = s["harness"]
        rows.append({
            "offered_qps": rate,
            "achieved_qps": h["achieved_qps"],
            "ratio": h["achieved_over_offered"],
            "ttft_p50_ms": s["ttft_ms"]["p50"],
            "ttft_p99_ms": s["ttft_ms"]["p99"],
            "tpot_p50_ms": s["tpot_ms"]["p50"],
            "goodput_req_s": s["goodput_req_s"],
            "lag_p99_ms": h["loadgen_lag_ms"]["p99"],
            "lag_drift": h["loadgen_lag_ms"]["drift"],
            "ttft_drift_frac": h.get("ttft_drift_frac"),
            "verdict": h["verdict"],
        })
        print(f"  qps={rate:<6g} verdict={h['verdict']:<14} "
              f"ttft_p99={s['ttft_ms']['p99']:.0f}ms")

    print("\nE0 · saturation sweep")
    table(rows, [("offered_qps", "offered"), ("achieved_qps", "achieved"),
                 ("ratio", "ach/off"), ("ttft_p50_ms", "TTFT p50"),
                 ("ttft_p99_ms", "TTFT p99"), ("goodput_req_s", "goodput"),
                 ("ttft_drift_frac", "queue growth"), ("verdict", "verdict")])

    valid = [r for r in rows if r["verdict"] != "client-bound"]
    sat = [r for r in valid if r["verdict"] == "saturated"]
    print()
    if not valid:
        print("  ALL RATES CLIENT-BOUND. Nothing here is a server measurement.")
    elif sat:
        first = min(sat, key=lambda r: r["offered_qps"])
        peak = max(valid, key=lambda r: r["achieved_qps"] or 0)
        print(f"  saturation begins at ~{first['offered_qps']:g} QPS offered")
        print(f"  peak achieved        {peak['achieved_qps']:.2f} req/s "
              f"(TTFT p99 {peak['ttft_p99_ms']:.0f} ms)")
        print("  -> report throughput only at or above the saturation point")
    else:
        print("  Never saturated in this range. Raise --rates; the top rate here is a "
              "floor on capacity, not a measurement of it.")

    if any(r["verdict"] == "client-bound" for r in rows):
        print("  NOTE: some rates were client-bound and are excluded. Shard the load "
              "generator across processes before reporting those rates.")

    write_sweep(args.out, tag, rows,
                {"experiment": "e0-saturation", "arm": arm.id, "rates": rates,
                 "image": [args.width, args.height], "n_per_rate": args.n})


if __name__ == "__main__":
    asyncio.run(main())
