# Track B · VLMs in real-time computer vision

!!! info "A second track, started 2026-09-24"
    The experiments before this page (E0–E7d) ask how fast and cheaply one VLM
    serves on one GPU. Track B asks what they led to: **when and how should a
    real-time computer-vision pipeline call a VLM, and what does it cost?**

## Two loops at two rates

| loop | rate | component | job |
|---|---|---|---|
| perception | every frame | detector + tracker (DeepStream / TensorRT) | *when* something changes and *where* |
| reasoning | per event | VLM | *what* it means, in open language |

The VLM is never real-time per frame and does not need to be: the system is
real-time because answers arrive fresh (≥ 95% under 2 s). What connects the loops
is a cost model the cascade experiments measured piece by piece:

**cameras per GPU ≈ VLM capacity ÷ (events per second × tokens per event)**

- the **gate** sets events per second — [E7](e7-cascade.md), [E7c](e7c-deepstream.md)
- the **crop** sets tokens per event — R1, next
- the **model** sets capacity and accuracy — [E7d](e7d-models.md)

## What Track B has to show

That the recipe *predicts*, including when **not** to crop and when **not** to call
a VLM, in domains with different properties:

| id | question | domain |
|---|---|---|
| R1 | the actor-crop reference level: margin, size, a marked frame, crop + full frame | surveillance (MEVA) |
| R2 | does it transfer where people are large and stations fixed? | manufacturing |
| R3 | a whole-scene domain, where the recipe should say "don't crop" | traffic |
| R4 | a trained action detector on the same clips | MEVA |
| R5 | AVA and a second model family | AVA |
| R6 | more frames of the crop instead of more pixels | MEVA |

**Where it does not apply:** hard real-time control loops (millisecond latency),
tasks where every frame counts (counting, metrology), and closed label sets at high
volume, where a trained model is cheaper and exact. Defect inspection is one: the
best model on [MMAD](https://arxiv.org/abs/2410.09453) reaches 74.9%, short of
industrial requirements.

## How it is run

Every idea enters a public [backlog](https://github.com/sadbodhs/vlm_inference_optimization/blob/main/BACKLOG.md);
each experiment is pre-registered in [RECIPE.md](https://github.com/sadbodhs/vlm_inference_optimization/blob/main/RECIPE.md)
before it runs; prior work is logged in [LITERATURE.md](https://github.com/sadbodhs/vlm_inference_optimization/blob/main/LITERATURE.md).
Exploratory checks may motivate an experiment but never count as its result.
