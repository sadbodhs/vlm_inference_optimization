# The per-task frontier

**Question:** does "how many vision tokens do I need" have one answer, or one per
task?

The same budget sweep on four tasks, each scored with its own metric — ANLS,
relaxed accuracy, exact match, exact match. Absolute scores are therefore **not
comparable across columns**; only the shape of each curve is. 400 samples per point,
Qwen2.5-VL-7B-AWQ on vLLM (`B0_vllm_awq_clean`, all caches off).

| budget (px) | DocVQA | ChartQA | TextVQA | VQAv2 |
|---|---|---|---|---|
| 50,176 | 0.293 · 101 tok | 0.297 · 101 tok | 0.530 · 95 tok | **0.815** · 93 tok |
| 200,704 | 0.767 · 287 | **0.873** · 287 | 0.792 · 281 | 0.877 · 276 |
| 451,584 | 0.909 · 600 | 0.892 · 577 | 0.860 · 590 | 0.892 · 385 |
| 802,816 | 0.943 · 1,034 | 0.895 · 629 | 0.873 · 977 | 0.892 · 385 |
| 1,605,632 | 0.944 · 1,999 | 0.897 · 638 | 0.880 · 1,029 | 0.892 · 385 |

Token counts differ per task because a budget is a *ceiling*: an image smaller than
the budget is sent at native size. Every VQAv2 image (COCO, median 273k px, max
410k px) is below 451,584 px, so the top three VQAv2 rows are the same input —
confirmed by identical per-sample scores on all 400.

![per-task frontier](img/task-frontier.png){ width="620" }

Each curve is normalised to its **own** peak, because the metrics are not comparable
in absolute terms — only the shape is.

## Saturation is task-dependent

| task | saturates at | peak |
|---|---|---|
| **VQAv2** | **≤ 385 tokens** (native size) | 0.892 |
| ChartQA | 577 tokens (202 ms) | 0.897 |
| TextVQA | 977 tokens (334 ms) | 0.880 |
| DocVQA | 1,034 tokens (338 ms) | 0.944 |

Fraction of each task's **own peak** retained, bootstrapped (paired on sample id):

| task | at ~100 tokens | at ~280 tokens |
|---|---|---|
| **VQAv2** | **91.3%** [87.3, 95.3] | **98.3%** [95.8, 100.8] |
| TextVQA | 60.2% [54.8, 65.9] | 90.1% [85.5, 94.6] |
| ChartQA | ~33% | 97.2% [93.6, 100.8] |
| DocVQA | ~31% | 81.3% [77.3, 85.1] |

At ~100 tokens the four tasks fall into three clearly separated groups: appearance (VQAv2) keeps nine-tenths of its accuracy, scene text keeps
three-fifths, and documents and charts keep under a third.

## The prediction, the correction, and the confirmation

The original hypothesis was a document-to-natural-image gradient. The first run
tested it with TextVQA as the "natural image" arm and got the order wrong —
**ChartQA cheapest, TextVQA middle, DocVQA most demanding.** TextVQA is text in
natural scenes: still a reading task. Choosing it was a category error.

The correction was that what governs the budget is **the angular size of the
smallest feature that carries the answer**:

- **objects, colours, counts, scenes** (VQAv2) — large features, survive almost any downscale
- **charts** — bars, lines, axis labels: large marks, need a modest budget to saturate
- **scene text** — signs and labels: mid-sized
- **document text** — dense small glyphs: the most demanding

VQAv2 is the test that rule predicted and had not yet faced, and it holds at the
extreme: with no text anywhere, **93 tokens keeps 91% of peak** — about 45 ms TTFT
against 136–158 ms at native size.

The same result under the official VQA metric (`min(#annotators agreeing / 3, 1)`,
stricter than the frozen any-of-ten exact-match scorer): peak 0.826, **89.1%
[85.0, 93.3]** retained at 93 tokens. The conclusion does not depend on the scorer.

## What this means in practice

For appearance questions — "is there a person", "what colour is the car", "what is
the man doing" — the default `max_pixels` of 1.6 MP is spent on resolution the
answer does not need. A 50k-pixel budget costs ~9 points of accuracy for a roughly
3× TTFT cut; a 200k budget costs ~1.5 points for a ~1.5× cut. For reading tasks the
same 50k budget throws away two-thirds of the accuracy. **The budget is a per-task
decision, and it should be set from what the question needs to see.**

!!! note "Caveats"
    - **VQAv2's ceiling is the dataset, not the model.** COCO images are small, so
      the curve cannot be pushed past 385 tokens; "saturates at ≤ 385" is a bound.
    - **A TTFT artefact on identical inputs.** The three native-size VQAv2 runs are
      byte-identical, yet the first ran ~20 ms slower across the whole distribution
      (p10 119 vs 99 ms; p50 158 vs 137 ms). The likely cause is per-image-shape
      warm-up inside the vision encoder, paid only the first time a shape is seen —
      *unverified*. It matters only where two budgets produce the same shapes, which
      in this sweep is only VQAv2. TTFT differences under ~15% between such points
      should not be read into.
    - **One model.** The ordering is measured on Qwen2.5-VL-7B only.
    - **The first VQAv2 run scored 0.000 at every budget** — a manifest bug, not a
      result. See [corrections](corrections.md).
