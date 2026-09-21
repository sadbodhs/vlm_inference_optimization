"""Aggregation, SLO goodput, and the R2 saturation verdict."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .result import RequestResult

# RTX 3090. Used for the decode roofline; see docs/e1.
HBM_BW_GB_S = 936.0


def pct(xs: list[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


@dataclass
class SLO:
    ttft_ms: float = 1000.0
    tpot_ms: float = 50.0


def summarize(
    results: list[RequestResult],
    *,
    offered_qps: float | None = None,
    slo: SLO | None = None,
) -> dict:
    slo = slo or SLO()
    ok = [r for r in results if r.ok]
    n_fail = len(results) - len(ok)

    ttft = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    tpot = [r.tpot_ms for r in ok if r.tpot_ms is not None]
    e2e = [r.e2e_ms for r in ok if r.e2e_ms is not None]
    lag = [r.loadgen_lag_ms for r in results]

    wall_s = None
    if ok:
        start = min(r.sent_s for r in ok)
        end = max(r.last_token_s or r.sent_s for r in ok)
        wall_s = max(end - start, 1e-9)

    out_tokens = sum(r.completion_tokens or len(r.itl_ms) + 1 for r in ok)

    met = [
        r
        for r in ok
        if (r.ttft_ms or 1e9) <= slo.ttft_ms and (r.tpot_ms or 1e9) <= slo.tpot_ms
    ]

    summary = {
        "n_requests": len(results),
        "n_ok": len(ok),
        "n_failed": n_fail,
        "wall_s": wall_s,
        "ttft_ms": {"p50": pct(ttft, 50), "p95": pct(ttft, 95), "p99": pct(ttft, 99),
                     "mean": float(np.mean(ttft)) if ttft else None},
        "tpot_ms": {"p50": pct(tpot, 50), "p95": pct(tpot, 95),
                     "mean": float(np.mean(tpot)) if tpot else None},
        "e2e_ms": {"p50": pct(e2e, 50), "p95": pct(e2e, 95)},
        "throughput_req_s": (len(ok) / wall_s) if wall_s else None,
        "output_tok_s": (out_tokens / wall_s) if wall_s else None,
        "goodput_req_s": (len(met) / wall_s) if wall_s else None,
        "slo": {"ttft_ms": slo.ttft_ms, "tpot_ms": slo.tpot_ms,
                 "fraction_met": (len(met) / len(ok)) if ok else None},
        "prompt_tokens": {
            "mean": float(np.mean([r.prompt_tokens for r in ok if r.prompt_tokens]))
            if any(r.prompt_tokens for r in ok) else None
        },
        "harness": harness_health(results, offered_qps, wall_s),
    }
    return summary


def harness_health(
    results: list[RequestResult], offered_qps: float | None, wall_s: float | None
) -> dict:
    """R2. Did we actually saturate the server, and was the client keeping up?

    Two distinct failure modes that look identical in a throughput number:

      * client-bound -- the load generator could not issue requests on schedule.
        loadgen_lag climbs monotonically. Any throughput reported here is the
        harness's, not the server's.
      * not-saturated -- the client kept up fine and the server never queued.
        The number is real, but it is a latency measurement wearing a throughput hat.

    Saturation is judged on queue growth, not on achieved/offered. With a Poisson
    arrival process the *realized* rate over n requests has standard error ~1/sqrt(n)
    -- at n=32 that is ±18%, enough to declare saturation from sampling noise alone.
    So the ratio is computed against the realized schedule (removing that variance)
    and the primary signal is whether TTFT drifts upward across the run, which is
    what a growing queue actually looks like.
    """
    ok = [r for r in results if r.ok]
    lag = [r.loadgen_lag_ms for r in results]
    if not lag:
        return {"verdict": "no-data"}

    lag_p99 = pct(lag, 99) or 0.0
    half = max(len(lag) // 2, 1)
    lag_drift = float(np.mean(lag[half:]) - np.mean(lag[:half])) if len(lag) > 3 else 0.0

    # Realized arrival rate: same denominator basis as the completion rate, so
    # Poisson variance cancels instead of masquerading as saturation.
    sched = sorted(r.scheduled_s for r in results)
    span = sched[-1] - sched[0]
    offered_realized = (len(sched) / span) if span > 0 else None
    achieved = (len(ok) / wall_s) if wall_s else None
    ratio = (achieved / offered_realized) if (achieved and offered_realized) else None

    # Queue growth: TTFT in the back half vs the front half of the run.
    ttft = [r.ttft_ms for r in sorted(ok, key=lambda r: r.sent_s) if r.ttft_ms is not None]
    ttft_drift = None
    ttft_drift_frac = None
    if len(ttft) >= 8:
        h = len(ttft) // 2
        front, back = float(np.mean(ttft[:h])), float(np.mean(ttft[h:]))
        ttft_drift = back - front
        ttft_drift_frac = ttft_drift / max(front, 1e-6)

    queue_growing = bool(ttft_drift_frac is not None and ttft_drift_frac > 0.25)

    if lag_p99 > 50.0 and lag_drift > 25.0:
        verdict = "client-bound"
        note = ("Load generator fell behind schedule and kept falling behind. "
                "Throughput here measures the harness. Shard the client or lower the rate.")
    elif queue_growing or (ratio is not None and ratio < 0.85):
        verdict = "saturated"
        note = ("Queue is growing: TTFT climbs across the run and/or completions trail "
                "arrivals. This is the regime throughput numbers are valid in.")
    else:
        verdict = "not-saturated"
        note = ("Client kept up and the server absorbed everything without a growing "
                "queue. Latency numbers are valid; throughput is a floor, not a capacity.")

    return {
        "verdict": verdict,
        "note": note,
        "loadgen_lag_ms": {"p50": pct(lag, 50), "p99": lag_p99, "drift": lag_drift},
        "ttft_drift_ms": ttft_drift,
        "ttft_drift_frac": ttft_drift_frac,
        "offered_qps": offered_qps,
        "offered_qps_realized": offered_realized,
        "achieved_qps": achieved,
        "achieved_over_offered": ratio,
    }


def decode_roofline_tok_s(weight_bytes: float, bw_gb_s: float = HBM_BW_GB_S) -> float:
    """Single-stream decode ceiling: one full pass over the weights per token."""
    return (bw_gb_s * 1e9) / weight_bytes
