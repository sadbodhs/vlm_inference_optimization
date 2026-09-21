# Results — Qwen2.5-VL-7B-AWQ on one RTX 3090

All numbers from `B0_vllm_awq_clean`: vLLM v0.29.0
(`sha256:c2914767…`), weights pinned at `536a3579`, prefix caching / chunked
prefill / multimodal cache **off**. Harness and server both in Docker, digests in
every `meta.json`.

---

## E1 · Decode is already at the physical limit

| measured | roofline | % of ceiling |
|---|---|---|
| **149.3 tok/s** | 161.2 tok/s | **92.7%** |

Single-stream decode re-reads the LLM weights once per token, so 936 GB/s ÷ 5.81 GB
is a hard ceiling. Hitting 92.7% of it means **there is nothing left to win at batch
1** — no kernel, scheduler or quantisation change can buy more than 8%. Every
remaining opportunity on this card is in batching or in prefill.

> The roofline must exclude the vision tower. It is 1.35 GB of bf16 weights that run
> once in prefill and are never touched again while tokens stream. Counting them
> gives a 131 tok/s ceiling, against which 149.3 tok/s reads as 114% — impossible,
> and a full day of hunting a harness bug that does not exist.

## E2 · What a vision token costs

```
TTFT = 17.5 ms + 319 ms per 1,000 vision tokens          (R² = 0.9994)
```

Measured at concurrency 1 across 66 → 2,027 vision tokens (43 → 669 ms). Two
consequences:

- **0.32 ms per vision token.** That is the exchange rate between image resolution
  and time-to-first-token.
- **A 17.5 ms floor** that no token reduction can ever recover. At document
  resolution it is 2.6% of TTFT — so at the resolutions that matter, TTFT *is*
  vision tokens, and almost nothing else.

## E3 · The accuracy–latency frontier (DocVQA, 200 samples, ANLS)

| max_pixels | prompt tokens | TTFT p50 | ANLS |
|---|---|---|---|
| 200,704 | 288 | 130 ms | 0.784 |
| 451,584 | 598 | 232 ms | 0.903 |
| 802,816 | 1,034 | 370 ms | 0.915 |
| **1,605,632** | **1,994** | **706 ms** | **0.939** |
| 3,211,264 | 3,830 | 1,421 ms | 0.932 |

Peak ANLS 0.939 matches Qwen2.5-VL-7B's published DocVQA score, which is the
calibration check — the harness agrees with a known value, not just with itself.

**Past the peak, more pixels are strictly worse.** Doubling the budget costs
**+101% TTFT for −0.007 ANLS**. The generous default is not a safe choice; it is a
slower and slightly less accurate one.

**The trade is steep and cheap.**

| give up | budget | TTFT | vs peak |
|---|---|---|---|
| 2.4 pts | 802,816 | 370 ms | **48% faster** |
| 3.6 pts | 451,584 | 232 ms | **67% faster** |

**The cliff is at the bottom.** 0.903 → 0.784 between 598 and 288 tokens: a 12-point
collapse where the text stops being resolvable. For documents the usable floor is
around 600 vision tokens.

## E0 · Throughput keeps looking healthy after goodput dies

| offered | achieved | goodput @ SLO | TTFT p99 | verdict |
|---|---|---|---|---|
| 2 | 1.72 | 1.72 | 196 ms | not-saturated |
| 8 | 6.58 | **6.58** | 295 ms | not-saturated |
| 16 | 9.52 | 0.59 | 1,667 ms | saturated |
| 32 | 9.90 | **0.00** | 3,517 ms | saturated |
| 64 | 9.96 | **0.00** | 4,489 ms | saturated |

SLO: TTFT ≤ 1 s, TPOT ≤ 50 ms.

This is the result worth publishing. **Raw throughput plateaus at 9.96 req/s and
stays there** while goodput falls to *zero*. A benchmark reporting req/s would call
this a 10 req/s server. Every request on it misses SLO by 3.5 seconds.

**Usable capacity is 6.58 req/s. Reporting the raw ceiling overstates it by 51%.**

## The card is power-limited, not heat-limited

Under sustained load at 64 QPS: **349.2 W against a 350 W cap**, 66 °C, SM clock
floor 1785 MHz, `thermal_throttled: false`, `power_capped: true`.

So the constraint is the power budget, not cooling. Better airflow would buy
nothing here; raising the power limit might. Distinguishing these needs the
driver's own clock-event reasons — a clock-ratio test calls a healthy card
throttled, because a 3090 under full load boosts to ~1700 of its 2100 MHz maximum
with nothing wrong.

## What the defaults were hiding

vLLM v0.29.0 ships with prefix caching, chunked prefill and a 4 GB multimodal cache
**on**. With them enabled, E1 logged a 75.4% prefix-cache hit rate and 86.7% MM-cache
hit rate:

| | TTFT p50 | decode |
|---|---|---|
| defaults on | 14.98 ms | 149.4 tok/s |
| caches off | **45.62 ms** | 149.3 tok/s |

**3× on TTFT, nothing on decode** — exactly the signature of a prefill-side cache.
Treating the shipped defaults as a baseline and then reporting prefix caching as an
improvement would measure the same optimisation twice and credit it once.

---

## Not measured

- **Only one model, one arm.** No 3B control, no InternVL, no FP16 comparison — so
  nothing here separates "property of VLM serving" from "property of this
  checkpoint on this card".
- **One dataset.** ChartQA and a natural-image set are frozen but unrun. The claim
  that the token-budget cliff is task-dependent is still a hypothesis.
- **E0 used synthetic 448×448 images**, not documents. Real document loads have
  much larger prefills and will saturate earlier.
- **No prefix-caching arm yet.** The 3× TTFT gap above is the confound measured,
  not the optimisation measured under a realistic reuse workload.
- **Single run per point.** No repeats, so no error bars. Treat differences under
  ~5% as noise.
- **E2's law under-predicts real documents by 6–18%.** Synthetic images of equal
  token count are cheaper than real ones; the cause is unexplained.
- **No FP8, ever, on sm_86.** Out of scope for this rig.
