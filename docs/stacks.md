# vLLM vs SGLang

Same model, same weights snapshot (`536a3579`), same quantisation (`awq_marlin`),
same client-resized images — verified to produce prompt token counts matching **to
the decimal** on both engines (287.2 / 598.7 / 1032.3 / 1994.2 / 3809.6).

Both vendors' default-on optimisations disabled on both sides: prefix/radix
caching, chunked prefill, multimodal caching. KV capacity matched to within 0.14%
(213,746 vs 213,456 tokens).

## Speed — SGLang wins, conditionally

| | vLLM v0.29.0 | SGLang v0.5.20 |
|---|---|---|
| decode | 149.05 ± 0.08 tok/s | **152.62 ± 0.07** (+2.4%) |
| TTFT p50 | 46.1 ms | **34.9 ms** (−24%) |
| usable goodput | 0.45 req/s | **0.55 req/s** (+22%) |
| raw ceiling | 1.56 req/s | 1.56 req/s |
| cold start (CUDA graphs) | **4 s** | 125 s |

The TTFT laws **cross** rather than one dominating:

```
vLLM    17.3 ms + 325 ms/1k tokens     ← lower slope
SGLang   4.5 ms + 354 ms/1k tokens     ← lower floor
```

A 4× lower fixed floor against a 9% steeper slope puts the crossover near **440
vision tokens**. SGLang wins on small images, vLLM on large ones — at the largest
budget vLLM is faster outright, 1,386 ms vs 1,551 ms. A single "which is faster"
answer would be wrong.

## Cost — SGLang scores 3–5 ANLS points lower

| vision tokens | vLLM | SGLang | gap |
|---|---|---|---|
| 287 | 0.783 | 0.752 | +0.031 |
| 599 | 0.918 | 0.875 | +0.043 |
| 1,032 | 0.944 | 0.894 | +0.051 |
| 1,994 | 0.956 | 0.907 | +0.049 |
| 3,810 | 0.950 | 0.909 | +0.041 |

Inputs are identical and decoding is greedy. **74.4% of outputs match byte for
byte**; the 25.6% that differ show SGLang generating ~9% fewer tokens — `'43%'`
becomes `'43'`, `'Reynolds Tobacco Co.'` becomes `'Reynolds Tobacco'`.

Ruled out as a harness artefact: streamed and non-streamed output from the same
server match on 12/12 samples, so the client is not dropping a final token.

!!! warning "Unresolved — stated as a hypothesis, not a finding"
    The gap **tracks vision-token count** (0.031 → 0.043 → 0.051 at 287 / 599 /
    1,032 tokens), which is the shape of numerical divergence accumulating in the
    vision tower rather than random sampling noise.

    SGLang logs `Multimodal attention backend not set. Use triton_attn`, so the two
    stacks run the vision encoder on **different kernels**. Confirming that means
    pinning both to the same backend and re-running. Until then this is a measured
    difference with an unproven cause.

**The honest headline: SGLang is 2.4% faster at decode and 24% faster to first
token, and 3–5 ANLS points less accurate on DocVQA.** Publishing the speed half
alone would be the failure R3 exists to prevent.
