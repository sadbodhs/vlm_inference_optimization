# E0 · Capacity, and why throughput lies

**Question:** at what arrival rate does this arm stop keeping up — and was the
*client* the real bottleneck?

400 distinct DocVQA pages, one per request, a **disjoint slice per rate** — zero
input reuse. SLO: TTFT ≤ 1 s, TPOT ≤ 50 ms.

## Result

| offered | achieved | goodput @ SLO | TTFT p50 | TTFT p99 | verdict |
|---|---|---|---|---|---|
| 0.5 | 0.45 | 0.35 | 724 ms | 4,085 ms | not-saturated |
| 1.0 | 0.90 | **0.45** | 798 ms | 2,024 ms | not-saturated |
| 1.5 | 1.32 | 0.21 | 1,606 ms | 3,837 ms | saturated |
| 2.0 | 1.52 | 0.04 | 4,083 ms | 10,633 ms | saturated |
| 4.0 | **1.56** | **0.00** | 14,554 ms | 29,085 ms | saturated |

![E0](img/e0-saturation-B0_vllm_awq_clean.png){ width="620" }

**Raw ceiling 1.56 req/s. Usable capacity 0.45 req/s.** Reporting the raw number
overstates what the server can deliver under SLO by **3.6×** — throughput keeps
reading 1.56 req/s while every request on it is 14 seconds late.

## Synthetic images overstate capacity by 6×

The same experiment on synthetic 448×448 squares gives a **9.96 req/s** ceiling.
Those carry ~256 vision tokens; a DocVQA page carries ~2,000, and prefill is what
saturates this server.

An image benchmark that does not use the real document size is measuring a workload
nobody runs — here by a factor of **6.4**.

## Caching buys nothing when inputs do not repeat

Same sweep, both arms, zero reuse — achieved throughput ratio (defaults ÷ clean)
is 1.00× at every rate, and both saturate at 1.56 req/s.

!!! danger "Getting here took three corrections"
    A 32-document pool cycled across requests made the defaults arm look **8×
    faster**. Fixing that left all five rates sharing one 80-document pool, so only
    the first rate was cold — visible as the two arms agreeing to within **3 ms at
    0.5 QPS** while every later rate was 13× apart.

    Reuse is now a variable the experiment sets and records, not a side effect of
    sweep order.

!!! note "This is a verbose-output workload"
    E0 does not apply E3's "answer using a single word or phrase" instruction, so
    the model generates full sentences out to `max_tokens=64`. Consistent across
    arms, so comparisons hold — but the absolute number does not transfer to
    short-span extraction, which would see higher capacity.
