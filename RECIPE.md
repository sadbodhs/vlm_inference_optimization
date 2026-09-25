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
| R1 | the actor-crop reference level: margin, size, marked frame, crop + full frame | surveillance (MEVA) | pre-registered 2026-09-25 |
| R2 | does R1's recipe transfer where people are large and stations fixed? | manufacturing (HA4M / InHARD) | backlog |
| R3 | a whole-scene domain, where the recipe should say "don't crop" | traffic | backlog |
| R4 | a trained action detector on the same clips: accuracy and cost | MEVA | backlog |
| R5 | AVA subset and a second model family | AVA | backlog |
| R6 | more frames of the crop instead of more pixels | MEVA | backlog |

## R1 · The actor-crop reference level — pre-registered 2026-09-25, before any measurement

**Question.** E7's ROI crop was one box around everyone, and never tested a clear
view of the actor. Centred on the people doing an action, at what margin and what
size does a zero-shot VLM recognise it best — and does that beat the full frame?

Exploratory evidence that motivated it (not a result): E7's crop covered 25–62% of
the frame with > 3 people in 73–97% of crops, enlarging the actor only 1.2–2.0×
(`results/e7e/cropsize.json`); Context-Aware RCNN on AVA found crop gains shrink
with actor size (+4.0 → −0.2 mAP) and peak at a 1.5–2.0× box expansion.

### Data and unit

- The 24 E7 clips. **Positives:** every MEVA activity with a person actor (499), at
  the 2 s window containing its midpoint, using the same two frames E7 sends
  (offsets 18 and 48). The **actor box** is the union of the activity's annotated
  actors (people and any vehicle) over both frames, from MEVA's boxes.
- **Negatives:** up to 499 DeepStream-tracked people, one per window, sampled
  (seed 0) from windows where they overlap no annotated actor of any activity. Their
  box is the tracker's.
- Recognised = the activity's group letter is in the answer. False alarm = any
  letter on a negative. **Lift** = recognised − chance, chance from shuffling each
  arm's answers across all its samples (20 shuffles), as in E7.

### Arms

Every arm sends two frames one second apart, capped at 451,584 px each, 16 answer
tokens, temperature 0. Crops share one prompt: *"close-up crops … centred on one
person or group; what is happening in the centre?"* with E7's eight options.

| arm | what the VLM sees |
|---|---|
| A0 | the full frame, E7's prompt (baseline) |
| A1 | the full frame with a red box drawn around the actor box; prompt asks about the boxed people |
| A2 | actor crop at **native** resolution, the box expanded **1.2× / 1.5× / 2× / 3×** |
| A3 | the 2× crop resized so the actor box is **112 / 224 / 448 px** tall |
| A4 | the 2× native crop (two frames) **plus** the full frame downscaled to ≤ 112,896 px, one request |
| A5 | the 2× native crop from **tracker** boxes matched to the actors (IoU ≥ 0.3); unmatched positives recorded as missed |

**Models:** Qwen3-VL-4B (E_q3vl_4b) on every arm; Qwen3-VL-8B (E_q3vl_8b) on A0,
A1, A2 at 2× and A4. vLLM alone on the card, util 0.80, E7d's settings.

### Predictions

- **R1.1** Lift peaks at a margin of **1.5× or 2×**; 1.2× and 3× are both lower than
  that peak, and the peak is above A0.
- **R1.2** Lift rises from 112 to 224 px actor height and changes by **< 3 points**
  from 224 to 448 px.
- **R1.3** The best A2 arm's gain over A0 is **larger for actors < 100 px tall** than
  for actors ≥ 100 px.
- **R1.4** On person + vehicle and doorway activities (groups A and C), **A4 ≥ the 2×
  A2 arm**.
- **R1.5** **A1 ≥ A0**, at < 5% more prompt tokens.
- **R1.6** The 2× native crop costs **≤ 50%** of A0's prompt tokens on average.
- **R1.7** The tracker matches the actors in **≥ 60%** of positives, and on those,
  A5's lift is within **5 points** of the ground-truth 2× crop.

A prediction holds on its point estimate; every comparison is also reported with a
95% paired bootstrap interval over clips. With ~500 positives, differences under
~8–10 points will not be separable, and the page will say so.

### Not measured

Live cameras (R1 is offline; the live run follows only for arms that keep
recognition); more than two frames (R6); other model families (R5); crops made
inside DeepStream (backlog 7).
