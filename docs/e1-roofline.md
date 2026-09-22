# E1 · Decode is already at the physical limit

**Question:** does single-stream decode land where 936 GB/s says it must?

E1 runs first in every sweep because it validates the instrument. Decode re-reads
the weights once per token, so bandwidth ÷ weight bytes is a hard ceiling. A
measured rate *above* it is physically impossible — it can only mean the harness is
wrong. If E1 fails that check, the runner refuses to run anything else.

## Result

| arm | measured | roofline | % of ceiling |
|---|---|---|---|
| vLLM (clean) | **149.05 ± 0.08** tok/s | 161.2 | **92.5%** |
| SGLang (clean) | **152.62 ± 0.07** tok/s | 161.2 | **94.7%** |

![E1](img/e1-roofline-B0_vllm_awq_clean.png){ width="520" }

Three independent runs per arm; run-to-run spread is **0.11–0.14%**, which is what
lets a 2.4% difference between stacks be called real rather than noise.

## There is nothing left to win at batch 1

At 92.5% of the ceiling, no kernel, scheduler or quantisation change can buy more
than 7.5% — and SGLang has already taken a quarter of it. Every remaining
opportunity on this card is in batching or in prefill, not in decode.

!!! danger "The roofline must exclude the vision tower"
    Decode re-reads only the **LLM** weights. The 675M-parameter bf16 vision tower
    runs once during prefill and is never touched again while tokens stream.

    Counting it gives 7.16 GB and a 131 tok/s ceiling, against which the measured
    149.4 tok/s reads as **114% — impossible**. The gate would have fired on a
    modelling error and sent us hunting a code bug that does not exist.

    Arms carry `decode_weight_bytes` separately, taken from the server's own load
    report rather than estimated.
