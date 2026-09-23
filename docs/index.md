# What does a VLM cost to serve?

*One topic on [Sadbodh](https://sadbodhs.github.io/) — an open bench for deep
learning systems. See also the [CV inference serving
study](https://sadbodhs.github.io/computer_vision_optimization/).*

Two questions, measured on one RTX 3090:

1. **What does a vision-language model actually cost to serve?**
2. **What does making it cheaper cost you in accuracy?**

Sibling study to [computer_vision_optimization](https://github.com/sadbodhs/computer_vision_optimization),
which asked the same kind of question about YOLO serving stacks.

## Why this is not the CV study with a bigger model

The YOLO study had one bottleneck axis and constant quality. A VLM breaks that:

**Two engines with opposite needs.** A vision encoder (compute-bound, fixed cost)
feeding an autoregressive decoder (prefill compute-bound, decode *bandwidth*-bound).

**Input size is a free variable.** An image is not one frame, it is *N tokens*, and
N is a policy decision — resolution, tiling, pruning. Nothing in the CV study had
this knob.

**Quality is a dependent variable.** Every token-reduction and quantisation trick
trades accuracy, so the deliverable is a frontier, not a ranking.

## What came out

| | |
|---|---|
| Decode is near the physical limit | **149.1 tok/s = 88.7%** of the 936 GB/s roofline |
| A vision token costs | **0.32 ms** of TTFT, over a 17.5 ms floor (R² = 0.9995) |
| DocVQA accuracy saturates | at **~1,000 vision tokens**; 4× beyond buys +0.011 ANLS |
| Throughput lies after saturation | raw ceiling **1.56 req/s**, usable **0.45 req/s** |
| Live video | **8 RTSP streams** at 1 fps inside a 2 s freshness budget |
| Past that knee | throughput **rises 70%** while freshness drops to **zero** |
| Synthetic images | overstate capacity by **6.4×** versus real document pages |
| Caching is a workload property | **3×** on repeated inputs, **1.00×** on distinct ones |

## What to read next

- [Methodology](methodology.md) — how the harness makes a wrong number loud
- [E2 · what a vision token costs](e2-vision-tokens.md) — the law the rest hangs off
- [E3 · the accuracy frontier](e3-token-budget.md) — the thesis experiment
- [Qwen2.5 vs Qwen3](models.md) — a generational comparison, and a prediction that was wrong
- [Corrections](corrections.md) — seven measurement bugs and what each would have published
- [Not measured](not-measured.md) — what these numbers do **not** entitle you to claim
