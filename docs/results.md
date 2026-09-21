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

## E3 · The accuracy–latency frontier (DocVQA, 200-sample subset, n=100 scored, ANLS)

| max_pixels | prompt tokens | TTFT p50 | ANLS | 95% CI |
|---|---|---|---|---|
| 200,704 | 288 | 130 ms | 0.784 | [0.713, 0.851] |
| 451,584 | 598 | **232 ms** | 0.903 | [0.848, 0.951] |
| 802,816 | 1,034 | 370 ms | 0.915 | [0.863, 0.960] |
| 1,605,632 | 1,994 | 706 ms | 0.939 | [0.895, 0.976] |
| 3,211,264 | 3,830 | 1,421 ms | 0.932 | [0.884, 0.973] |

The top value, 0.939, is consistent with Qwen2.5-VL-7B's published DocVQA score.
That is the calibration check: the harness agrees with a known external value, not
only with itself.

### What n=100 does and does not support

Bootstrapped paired differences (same documents at both budgets, 10k resamples):

| A | B | Δ | 95% CI | |
|---|---|---|---|---|
| 200,704 | 451,584 | −0.120 | [−0.192, −0.051] | **different** |
| 200,704 | 1,605,632 | −0.156 | [−0.229, −0.088] | **different** |
| 451,584 | 1,605,632 | −0.036 | [−0.083, +0.006] | indistinguishable |
| 802,816 | 1,605,632 | −0.024 | [−0.059, +0.005] | indistinguishable |
| 1,605,632 | 3,211,264 | +0.007 | [−0.013, +0.033] | indistinguishable |

**Above ~600 vision tokens, accuracy is flat within measurement error while TTFT
grows 6×** — 232 ms to 1,421 ms across budgets whose accuracy cannot be told apart.
The practical consequence is strong and the statistical claim is weak, and they point
the same way: run at 451,584 and save 67% of TTFT for no *measurable* accuracy cost.

**The cliff at the bottom is real.** 288 tokens loses 0.12–0.16 ANLS against every
larger budget, with confidence intervals well clear of zero. For documents the usable
floor is somewhere between 288 and 598 vision tokens — this sweep does not resolve
where.

> An earlier draft of this page claimed a peak at 1,605,632 and that more pixels past
> it were "strictly worse" (−0.007 ANLS). Neither survives the confidence interval:
> the four largest budgets are statistically identical, and −0.007 sits inside
> [−0.013, +0.033]. Resolving the top of the frontier needs n≈500–1000, not 100.

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
- **Single run per point.** No repeats, so no error bars on any *latency* number.
  Treat latency differences under ~5% as noise.
- **Accuracy resolves only large effects.** n=100 gives roughly ±0.05 ANLS at 95%,
  which is wider than every difference above 600 vision tokens. The shape of the top
  of the frontier is unmeasured, not flat-by-finding.
- **E2's law under-predicts real documents by 6–18%.** Synthetic images of equal
  token count are cheaper than real ones; the cause is unexplained.
- **No FP8, ever, on sm_86.** Out of scope for this rig.
