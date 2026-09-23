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
    # Synthetic squares understate capacity badly: a real page carries several
    # times the vision tokens, and prefill is what saturates this server.
    p.add_argument("--manifest", default=None,
                   help="use real documents instead of synthetic images")
    # Same reason as E3: asking the server to cap pixels works on vLLM and is
    # ignored by SGLang, so a cross-stack saturation comparison would be run at
    # different token counts per stack. Resizing before sending fixes the workload.
    p.add_argument("--resize", choices=["client", "server"], default="client")
    args = p.parse_args()

    from bench.data import load_manifest, synthetic
    from bench.harness import run_arm

    arm = resolve_arm(args)
    arm.max_tokens = args.max_tokens
    tag = args.tag or f"e0-saturation-{arm.id}"
    rates = [float(x) for x in args.rates.split(",")]
    # Every request gets its own document, and every rate its own slice.
    #
    # Cycling a small pool is harmless with all caches off, but with prefix/multimodal
    # caching on it turns the sweep into a ~100% reuse workload -- the best case for a
    # cache and a workload almost nobody has. A 32-sample pool made the defaults arm
    # look 8x faster than the clean one; sharing a single 80-sample pool across rates
    # then left only the FIRST rate cold, which showed up as that rate matching the
    # no-cache arm to within 3 ms while every later rate was 13x faster. Sharing one pool across the sweep makes the
    # first rate cold and every later rate a cache hit -- visible in the data as a
    # first rate that matches the no-cache arm exactly and later rates that do not.
    # Reuse must be a variable this experiment sets, not a side effect of sweep order.
    need = args.n * len(rates)
    if args.manifest:
        pool = load_manifest(args.manifest, limit=need)
        workload = args.manifest
        if len(pool) < need:
            print(f"  WARNING: manifest has {len(pool)} samples but this sweep needs "
                  f"{need} for zero reuse across rates. Later rates will re-send "
                  f"earlier documents and any enabled cache will serve them.")
    else:
        pool = synthetic(need, width=args.width, height=args.height)
        workload = f"synthetic {args.width}x{args.height}"
    if args.resize == "client" and arm.max_pixels:
        budget = arm.max_pixels
        arm.max_pixels = None            # stop asking the server to do it
        pool = [s.resized(budget) for s in pool]   # outside the timed path
        print(f"  client-side resize to <= {budget} px applied to {len(pool)} samples")

    reuse = max(0.0, 1 - len(pool) / need)
    print(f"workload: {workload}  ({len(pool)} distinct samples across {len(rates)} "
          f"rates x {args.n} requests -> reuse {reuse * 100:.0f}%)")

    rows = []
    for i, rate in enumerate(rates):
        # Disjoint slice per rate, so no rate benefits from a previous rate's cache.
        lo = (i * args.n) % max(len(pool), 1)
        samples = pool[lo:lo + args.n] or pool[: args.n]
        s = await run_arm(
            arm, samples, mode="open", rate_qps=rate,
            repeats=max(round(args.n / len(samples)), 1),
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
            "failed_frac": s["failed_frac"],
            # a rate with failed requests is not a capacity measurement: its latency
            # and goodput are computed over the survivors
            "verdict": h["verdict"] if s["valid"] else "failures",
        })
        print(f"  qps={rate:<6g} verdict={h['verdict']:<14} "
              f"ttft_p99={s['ttft_ms']['p99']:.0f}ms")

    print("\nE0 · saturation sweep")
    table(rows, [("offered_qps", "offered"), ("achieved_qps", "achieved"),
                 ("ratio", "ach/off"), ("ttft_p50_ms", "TTFT p50"),
                 ("ttft_p99_ms", "TTFT p99"), ("goodput_req_s", "goodput"),
                 ("ttft_drift_frac", "queue growth"), ("verdict", "verdict")])

    valid = [r for r in rows if r["verdict"] not in ("client-bound", "failures")]
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
                 "workload": workload, "image": [args.width, args.height],
                 "n_per_rate": args.n, "distinct_samples": len(pool),
                 "reuse_fraction": reuse, "resize": args.resize})


if __name__ == "__main__":
    asyncio.run(main())
