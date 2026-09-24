# Track B · VLMs in real-time computer vision: a measured recipe

Track A (`PLAN.md`, experiments E0–E7d) asks how fast and cheaply one VLM serves on
one GPU. Track B asks the question that grew out of its cascade experiments:

> **When and how should a real-time computer-vision pipeline call a VLM, and what
> does it cost?** Across domains, not only surveillance.

Both tracks share this repo's harness, Docker images, DeepStream pipeline and model
arms. Track A closes with E7d; new work lands here.

## The architecture under test: two loops at two rates

| loop | rate | component | job |
|---|---|---|---|
| perception | every frame | detector + tracker (DeepStream / TensorRT) | *when* something changes and *where* |
| reasoning | per event | VLM | *what* it means, in open language |

The system is real-time in the sense of **fresh answers** (E7b's bar: ≥ 95% under
2 s), not per-frame VLM inference. The working cost model, from E7–E7d:

    cameras per GPU  ≈  VLM capacity / (events per second per camera × tokens per event)

- the **gate** sets events per second (E7: 38–45% of windows on MEVA)
- the **crop** sets tokens per event (R1)
- the **model** sets capacity and accuracy (E7d)

The claim Track B has to earn is that this recipe *predicts* — including when not
to crop and when not to call a VLM at all — in domains with different properties.

## Process

1. Every new idea goes to [`BACKLOG.md`](BACKLOG.md) first, never straight into a
   running experiment.
2. Pick by value per GPU-hour; **pre-register here** (dated) before measuring.
3. Run under the shared GPU lock (`scripts/gpu_lock.sh`); one measurement on the
   card at a time.
4. Publish the page, the prediction scorecard and any corrections.
5. New findings return to the backlog.

Rules:

- **A running experiment's question is frozen.** Additions become a new R-number,
  or a dated amendment written before the affected data is looked at.
- **Exploratory analyses are labelled as such** (e.g. `experiments/e7e_cropsize.py`,
  `experiments/e7e_groups.py`). They may motivate an experiment; they are never its
  result.
- Prior work is logged in [`LITERATURE.md`](LITERATURE.md) with how it bears on us.

## Experiments

| id | question | domain | status |
|---|---|---|---|
| R1 | the actor-crop reference level: margin, size, marked frame, crop + full frame | surveillance (MEVA) | designing — predictions to be registered below before any run |
| R2 | does R1's recipe transfer where people are large and stations fixed? | manufacturing (HA4M / InHARD) | backlog |
| R3 | a whole-scene domain, where the recipe should say "don't crop" | traffic | backlog |
| R4 | a trained action detector on the same clips: accuracy and cost | MEVA | backlog |
| R5 | AVA subset and a second model family | AVA | backlog |
| R6 | more frames of the crop instead of more pixels | MEVA | backlog |

## R1 · The actor-crop reference level — *draft, not yet registered*

Exploratory evidence that motivated it (2026-09-24, not a result):

- E7's ROI crop was one box around **everyone**: 25–62% of the frame, > 3 people in
  73–97% of crops, the actor only 1.2–2.0× larger than in the full frame
  (`results/e7e/cropsize.json`). It never tested a clear view of the actor.
- Grouping tracked people by proximity (gap ≤ 1 body height) keeps 98% of tracked
  two-person activities in one crop; at native resolution group crops cost ~2.4×
  fewer tokens than the full frame (`results/e7e/groups_feasibility.json`).
- Context-Aware RCNN (AVA): crop gains shrink with actor size (+4.0 → −0.2 mAP) and
  peak at a 1.5–2.0× box expansion.

Arms, predictions and scoring will be written here and committed before the first
R1 request is sent.
