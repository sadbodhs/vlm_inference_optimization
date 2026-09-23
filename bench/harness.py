"""Run one arm against one dataset under one load profile, and write results.

    results/<run_id>/
        latency.jsonl   one row per request
        outputs.jsonl   generated text (kept apart so latency.jsonl stays small)
        summary.json    aggregates + SLO goodput + R2 saturation verdict
        meta.json       everything needed to reproduce, incl. what was NOT measured

Accuracy is scored on the outputs of the *same* run that produced the latencies.
Scoring a separate pass is the standard way to end up with a Pareto plot whose two
axes come from different experiments.
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx

from . import gpu, loadgen
from .arms import Arm
from .client import stream_chat
from .data import Sample
from .metrics import SLO, summarize
from .result import RequestResult
from .scorers import SCORERS


def _git_sha() -> str | None:
    """Commit that produced this run.

    git is not installed in the harness image, so shelling out inside the container
    silently returns None and every result loses its provenance. run_harness.sh
    resolves it on the host and passes it in; the subprocess call is the fallback
    for running outside a container.
    """
    if sha := os.environ.get("GIT_SHA"):
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


MAX_FAILED_FRAC = 0.01


async def run_arm(
    arm: Arm,
    samples: list[Sample],
    *,
    mode: str = "open",
    rate_qps: float = 1.0,
    concurrency: int = 1,
    repeats: int = 1,
    scorer: str | None = None,
    prompt_suffix: str = "",
    slo: SLO | None = None,
    results_root: str | Path = "results",
    run_id: str | None = None,
    unmeasured: list[str] | None = None,
    seed: int = 0,
) -> dict:
    run_id = run_id or f"{arm.id}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out_dir = Path(results_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    work = [samples[i % len(samples)] for i in range(len(samples) * repeats)]
    gpu_before = gpu.probe()
    extra = arm.request_extra()

    limits = httpx.Limits(max_connections=max(concurrency * 4, 256))
    async with httpx.AsyncClient(limits=limits) as http:

        async def send(i: int, scheduled_s: float) -> RequestResult:
            s = work[i]
            r = await stream_chat(
                http,
                base_url=arm.base_url,
                model=arm.model,
                messages=s.to_messages(prompt_suffix),
                arm_id=arm.id,
                sample_id=s.id,
                scheduled_s=scheduled_s,
                max_tokens=arm.max_tokens,
                temperature=arm.temperature,
                extra_body=extra,
            )
            r.image_px = s.image_px
            return r

        t0 = time.perf_counter()
        sampler = gpu.Sampler()
        sampler.__enter__()
        if mode == "open":
            results = await loadgen.open_loop(len(work), rate_qps, send, seed=seed)
        elif mode == "closed":
            results = await loadgen.closed_loop(len(work), concurrency, send)
        elif mode == "sequential":
            results = await loadgen.sequential(len(work), send)
        else:
            sampler.__exit__()
            raise ValueError(f"unknown mode {mode!r}")
        sampler.__exit__()
        wall = time.perf_counter() - t0

    gpu_after = gpu.probe()

    summary = summarize(
        results, offered_qps=rate_qps if mode == "open" else None, slo=slo
    )
    summary["mode"] = mode
    summary["wall_clock_s"] = wall

    # ---- accuracy, from this run's own outputs (R3) ----------------------
    by_id = {s.id: s for s in samples}
    if scorer:
        fn = SCORERS[scorer]
        scores = [
            fn(r.text, by_id[r.sample_id].answers)
            for r in results
            if r.ok and r.sample_id in by_id and by_id[r.sample_id].answers
        ]
        summary["accuracy"] = {
            "scorer": scorer,
            "n_scored": len(scores),
            "score": (sum(scores) / len(scores)) if scores else None,
        }
    else:
        summary["accuracy"] = {"scorer": None, "score": None,
                               "note": "not scored in this run"}

    with (out_dir / "latency.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_row()) + "\n")
    with (out_dir / "outputs.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps({"sample_id": r.sample_id, "ok": r.ok,
                                "text": r.text, "error": r.error}) + "\n")

    meta = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        # A dirty tree means the code that ran is not the code at that commit.
        "git_dirty": os.environ.get("GIT_DIRTY"),
        "arm": asdict(arm),
        "load": {"mode": mode, "rate_qps": rate_qps, "concurrency": concurrency,
                  "repeats": repeats, "n_requests": len(work), "seed": seed},
        # Part of the measurement, not a detail: DocVQA-style metrics score a bare
        # span, so the instruction that produces one belongs in the record.
        "prompt_suffix": prompt_suffix,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        # Container provenance. Every arm runs in Docker; these are injected by
        # docker/run_harness.sh and docker/run_vllm.sh so a result is traceable to
        # the exact images that produced it.
        "images": {
            "harness": os.environ.get("HARNESS_IMAGE"),
            "server": os.environ.get("SERVER_IMAGE"),
        },
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        # Sampled during the run, not inferred from before/after (see gpu.Sampler).
        "thermal": sampler.summary(),
        # R4: the things this run does not entitle you to claim.
        "not_measured": unmeasured or [],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    summary["run_id"] = run_id
    summary["out_dir"] = str(out_dir)

    # A run in which nothing succeeded must not return quietly: the tables would
    # render as empty columns and the sweep would continue producing files that
    # look like results. Fail here, with the reason the requests actually gave.
    if results and summary["n_ok"] == 0:
        first = next((r.error for r in results if r.error), "unknown")
        raise RuntimeError(
            f"arm {arm.id!r}: all {len(results)} requests failed against "
            f"{arm.base_url} -- first error: {first}"
        )
    # ...and a run in which SOME requests failed must not pass as a measurement
    # either. Latency percentiles and goodput are computed over the successful
    # requests, so failures silently improve them: an SGLang E0 rate at which 43%
    # of requests returned HTTP 500 read as that stack's BEST goodput, and was
    # published as "+22% over vLLM". Anything over 1% failed is flagged here and
    # disqualified by the sweeps (MAX_FAILED_FRAC).
    summary["failed_frac"] = (1 - summary["n_ok"] / len(results)) if results else 0.0
    summary["valid"] = summary["failed_frac"] <= MAX_FAILED_FRAC
    if not summary["valid"]:
        first = next((r.error for r in results if r.error), "unknown")
        print(f"  WARNING {run_id}: {summary['failed_frac']:.1%} of requests failed "
              f"-- this rate is not a measurement. First error: {first[:160]}")
    return summary


def run_arm_sync(*args, **kwargs) -> dict:
    return asyncio.run(run_arm(*args, **kwargs))
