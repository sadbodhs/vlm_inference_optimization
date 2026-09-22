# E4 · Live video, measured by staleness

**Question:** how many live RTSP streams can one 3090 understand, and how stale is
the answer?

TTFT is the wrong metric for live video. A camera does not wait for the model, so
what matters is:

```
staleness = answer_received − frame_captured
```

A server answering in 300 ms about a frame from 8 seconds ago is useless for
anything live, and every request-latency metric calls it healthy.

Decoders run continuously in their own containers, keeping only the newest frame,
so backlog is **dropped rather than queued** — which is what a live system should
do and what keeps staleness bounded by the model instead of growing without limit.

## Eight streams work

| streams | analyses/s | staleness p50 | staleness p95 | within 2 s |
|---|---|---|---|---|
| 1 | 1.00 | 652 ms | 723 ms | 100% |
| 2 | 2.00 | 635 ms | 1,368 ms | 100% |
| 4 | 4.00 | 1,078 ms | 1,561 ms | 100% |
| 8 | 8.00 | 1,500 ms | 1,888 ms | 100% |

## The ninth breaks everything

| demand /s | achieved /s | frames dropped | staleness p50 | fresh < 2 s |
|---|---|---|---|---|
| 8 | 8.00 | 0 | 1,467 ms | **98%** |
| 16 | **13.64** | 106 | 20,029 ms | **0%** |
| 64 | 13.91 | 2,254 | 26,152 ms | 0% |

Past the knee **throughput rises 70%** while the fraction of answers describing a
current frame goes to **zero**. Every answer past that point describes a frame
roughly 26 seconds old.

The inversion is real, not an artefact: more requests in flight means larger batches
and more total GPU work completed. The server is doing more — it is doing it about
the past. A throughput-only benchmark would report 13.9 analyses/s as capacity, a
74% *improvement* on the only rate that works.

**Usable capacity is 8 analyses/s:** eight streams at 1 fps, or one at 8 fps.

!!! danger "E4 was a closed loop and reported its own pacing as capacity"
    The first version awaited every analysis before the next tick, pinning
    concurrency at the stream count. It produced a flat 8.18 analyses/s across 8,
    16, 32 and 64 analyses/s of demand, which reads as clean saturation and is
    really 8 requests at ~980 ms each — arithmetic about the harness.

!!! note "No accuracy is claimed here"
    The stock clip is a 3D-animated short, so scene descriptions are of animated
    content. The pipeline numbers are real; nothing here supports a claim about
    scene-understanding quality on real cameras.
