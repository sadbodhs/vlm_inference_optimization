#!/usr/bin/env python3
"""E4 -- how many live RTSP streams can one 3090 understand, and how stale is the answer?

TTFT is the wrong metric for live video. A camera does not wait for the model, so
what matters is **staleness**: how old the frame is by the time its answer arrives.
A server that answers in 300 ms about a frame from 8 seconds ago is useless for
anything live, and every request-latency number would call it healthy.

    staleness = answer_received - frame_captured

Decoders run continuously and independently (docker/run_decoders.sh), exactly as in
a real deployment. Each analysis tick reads the NEWEST frame per stream, so a
backlog is dropped rather than queued -- which is what a live system should do, and
which makes staleness bounded by the model instead of growing without limit.

    docker/run_decoders.sh start 4
    python3 experiments/e4_video_stream.py --arm arms/B0_vllm_awq_clean.yaml --streams 4

Scene-understanding accuracy is NOT measured here. The stock clip is 3D animation,
so any accuracy claim would be about animated content. This measures the pipeline.
"""
from __future__ import annotations

import asyncio
import statistics as st
import time
from pathlib import Path

import httpx

from _common import base_parser, resolve_arm, table, write_sweep

TASKS = {
    "describe": "Describe this scene in one short sentence.",
    "presence": "Is there an animal visible in this image? Answer yes or no.",
    "count": "How many distinct objects are in the foreground? Answer with a number.",
    "change": "What is happening in this scene? Answer in under ten words.",
}


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--streams", type=int, default=1, help="how many cams to watch")
    p.add_argument("--frame-dir", default="/tmp/vlm_frames")
    p.add_argument("--task", default="describe", choices=sorted(TASKS))
    p.add_argument("--duration", type=float, default=60.0, help="seconds to run")
    p.add_argument("--interval", type=float, default=1.0,
                   help="seconds between analysis ticks per stream")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--staleness-slo", type=float, default=2000.0,
                   help="ms; an answer older than this is stale, not wrong")
    p.add_argument("--max-inflight", type=int, default=256,
                   help="cap on concurrent analyses; beyond this, frames are dropped")
    args = p.parse_args()

    from bench.client import stream_chat
    from bench.data import Sample
    from bench.video import read_latest, wait_for_streams
    from bench import gpu

    arm = resolve_arm(args)
    arm.max_tokens = args.max_tokens
    tag = args.tag or f"e4-video-{arm.id}-s{args.streams}"
    frame_dir = Path(args.frame_dir)
    question = TASKS[args.task]

    wanted = [f"cam{i}" for i in range(1, args.streams + 1)]
    live = wait_for_streams(frame_dir, wanted, timeout_s=60)
    if len(live) < len(wanted):
        missing = set(wanted) - set(live)
        print(f"  WARNING: no frames from {sorted(missing)} -- excluded. "
              f"A dead decoder would otherwise score perfect staleness.")
    if not live:
        raise SystemExit("no live streams; start decoders first")
    print(f"watching {len(live)} stream(s): {', '.join(live)}  task={args.task!r}")

    records: list[dict] = []
    sampler = gpu.Sampler()
    sampler.__enter__()
    t_end = time.time() + args.duration

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as http:

        async def analyse(stream: str) -> None:
            f = read_latest(frame_dir, stream)
            if f is None:
                records.append({"stream": stream, "ok": False, "error": "no frame"})
                return
            s = Sample(id=f"{stream}-{f.captured_s:.3f}", question=question,
                       image_b64=f.b64, image_mime="jpeg")
            r = await stream_chat(
                http, base_url=arm.base_url, model=arm.model,
                messages=s.to_messages(), arm_id=arm.id, sample_id=s.id,
                scheduled_s=time.perf_counter(), max_tokens=arm.max_tokens,
                temperature=arm.temperature, extra_body=arm.request_extra(),
            )
            done = time.time()
            records.append({
                "stream": stream,
                "ok": r.ok,
                "error": r.error,
                # The metric. Everything else is diagnostics.
                "staleness_ms": (done - f.captured_s) * 1e3,
                "frame_age_at_read_ms": f.age_at_read_ms,
                "ttft_ms": r.ttft_ms,
                "e2e_ms": r.e2e_ms,
                "prompt_tokens": r.prompt_tokens,
                "text": (r.text or "").strip()[:160],
            })

        # Open loop. Ticks fire on a fixed schedule whether or not the previous
        # tick finished, so demand is independent of the server's completion rate.
        #
        # The obvious `await gather(...)` per tick is a CLOSED loop: it pins
        # concurrency at the stream count and cannot offer more load than it
        # completes, so tightening the interval changes nothing and the measurement
        # reports its own pacing as the server's capacity. Measured here, that gave
        # an identical 8.18 analyses/s across an 8x range of demand.
        inflight: set[asyncio.Task] = set()
        tick = 0
        dropped = 0
        next_tick = time.time()
        while time.time() < t_end:
            for stream in live:
                if len(inflight) >= args.max_inflight:
                    # Shedding load is a result, not an error: a live system that
                    # cannot keep up should drop frames rather than grow a queue.
                    dropped += 1
                    continue
                t = asyncio.create_task(analyse(stream))
                inflight.add(t)
                t.add_done_callback(inflight.discard)
            tick += 1
            next_tick += args.interval
            slack = next_tick - time.time()
            if slack > 0:
                await asyncio.sleep(slack)
        if inflight:
            # Let what is already running finish, but do not wait forever on a
            # server that has fallen far behind.
            await asyncio.wait(inflight, timeout=30)

    sampler.__exit__()
    wall = args.duration

    ok = [r for r in records if r["ok"]]
    stale = [r["staleness_ms"] for r in ok]
    met = [x for x in stale if x <= args.staleness_slo]

    def pct(xs, q):
        return st.quantiles(xs, n=100)[q - 1] if len(xs) > 2 else (max(xs) if xs else None)

    row = {
        "streams": len(live),
        "ticks": tick,
        "dropped": dropped,
        "demand_per_s": len(live) / args.interval,
        "task": args.task,
        "interval_s": args.interval,
        "analyses": len(records),
        "ok": len(ok),
        "analyses_per_s": len(ok) / wall,
        "per_stream_per_s": len(ok) / wall / len(live),
        "staleness_p50_ms": st.median(stale) if stale else None,
        "staleness_p95_ms": pct(stale, 95),
        "staleness_max_ms": max(stale) if stale else None,
        "fresh_fraction": (len(met) / len(ok)) if ok else None,
        "ttft_p50_ms": st.median([r["ttft_ms"] for r in ok if r["ttft_ms"]]) if ok else None,
        "prompt_tokens": st.median([r["prompt_tokens"] for r in ok if r["prompt_tokens"]]) if ok else None,
    }

    print(f"\nE4 · {len(live)} stream(s), {args.task}, {args.duration:.0f}s")
    table([row], [("streams", "streams"), ("demand_per_s", "demand/s"),
                  ("analyses_per_s", "analyses/s"), ("dropped", "dropped"),
                  ("staleness_p50_ms", "stale p50"), ("staleness_p95_ms", "stale p95"),
                  ("fresh_fraction", f"<{args.staleness_slo:.0f}ms"),
                  ("prompt_tokens", "vis tok")])

    if ok:
        print(f"\n  sample answers:")
        for r in ok[:3]:
            print(f"    [{r['stream']}] {r['text']!r}")

    if row["fresh_fraction"] is not None and row["fresh_fraction"] < 0.95:
        print(f"\n  {100 * (1 - row['fresh_fraction']):.0f}% of answers described a frame "
              f"older than {args.staleness_slo:.0f} ms.")
        print("  For live video that is the failure mode, and no request-latency "
              "metric shows it.")

    write_sweep(args.out, tag, [row],
                {"experiment": "e4-video-stream", "arm": arm.id,
                 "streams": live, "task": args.task, "question": question,
                 "duration_s": args.duration, "interval_s": args.interval,
                 "staleness_slo_ms": args.staleness_slo,
                 "thermal": sampler.summary(),
                 "not_measured": ["scene-understanding accuracy (clip is 3D animation)",
                                  "decode cost (runs in separate containers)"]})


if __name__ == "__main__":
    asyncio.run(main())
