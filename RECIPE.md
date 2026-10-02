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
| R1 | the actor-crop reference level: margin, size, marked frame, crop + full frame | surveillance (MEVA) | published 2026-09-25: 6 of 7 held; see docs/r1-actor-crops.md |
| R1b | R1 without labels: proximity-group crops on gate-fired windows | surveillance (MEVA) | published 2026-09-25: 1 of 5 held; see docs/r1b-label-free.md |
| R2 | do hand crops beat the full frame where the answer is the part in the hand? | manufacturing (HA4M) | published 2026-09-26: 3 of 9 held; zero-shot floor; see docs/r2-manufacturing.md |
| R2b | does the crop hold more information for a trained head? (frozen-encoder probe) | manufacturing (HA4M) | published 2026-09-27: 3 of 5 held; full+wide +4.5 [+1.5,+7.5]; fine-tuning rule met |
| R1c | a precision pass: does a second look (or a stricter prompt) cut the false alarms crops bring? | surveillance (MEVA) | pre-registered 2026-10-02 |
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

## R1b · Label-free crops on gate-fired windows — pre-registered 2026-09-25, before any measurement

**Question.** R1 chose each crop from the annotated participants, which leaked who
was involved, and scored crops of people the gate might never send. With the crop
chosen by the tracker alone and scored the way the cascade runs (on the windows
E7c's gate fires), does centring on people still beat the full frame, and at what
cost in false alarms and tokens?

### Facts, measured before registering (`experiments/r1b_windows.py`, no VLM)

- `track-motion` fires on **1,531 of 3,600 windows** (42.5%).
- People tracked on the two VLM frames are grouped by proximity (gap ≤ one body
  height; a vehicle joins the nearest person within one body height). Fired windows
  hold **2.2 groups** on average (p90 5). 237 hold none (vehicle-only motion).
  44 groups beyond a cap of 7 per window are dropped.
- Label-free group crops (2× the group box) contain an actor of **424 of the 492
  person activities** that fall in a fired window (86%).

### Arms

All arms: the same two frames per fired window (offsets 18 and 48), 16 answer
tokens, frames capped at 451,584 px, context frames at 112,896 px. Windows with no
group fall back to the full frame in every arm.

| arm | per fired window |
|---|---|
| F | one request: the full frame, E7's prompt |
| M | one request: the full frame with every group's crop drawn in red; prompt about the boxed people |
| GS | one request **per group**: the group's 2× crop (R1's A2) |
| GSC | one request **per group**: the 2× crop plus the full frame at low resolution (R1's A4) |
| GO | **one** request: every group's crop (two frames each) plus one low-resolution full frame; prompt asks what happens in any close-up |

**Models:** Qwen3-VL-4B on every arm; Qwen3-VL-8B on F and GO.

### Scoring

As E7, per activity instance: recognised if its group letter is in the answer for
any fired window it overlaps (for GS and GSC, the union of that window's group
answers). Chance from shuffling window answers across fired windows (20 shuffles).
**False alarms per hour** = letters absent from the window's labels, over the 2 h of
video. **Tokens per fired window** = the sum over that window's requests. Arms are
compared to F by a paired bootstrap over clips.

### Predictions (Qwen3-VL-4B unless stated)

- **R1b.1** The gain survives without labels: GSC's lift exceeds F's by **≥ 5
  points**.
- **R1b.2** Context tames the false alarms: GS's false alarms per hour are **≥ 2×** F's;
  GSC's are **≤ 1.5×** F's.
- **R1b.3** One request per window is the deployable form: GO's lift is **within 5
  points** of GSC's, at **≤ 60%** of GSC's tokens per window and **≤** F's.
- **R1b.4** Marking without the leak: M's lift is **≥** F's, its false alarms per hour
  within **±25%** of F's, at **< 5%** more tokens.
- **R1b.5** Qwen3-VL-8B gains **less** from GO over F than the 4B does (it starts
  higher: +40 vs +26 in R1).

### Not measured

Live cameras; other gates; group-size or grouping-distance sweeps; answers
attributed to a specific group (reported, not predicted).

## R1c · A precision pass — pre-registered 2026-10-02, before any measurement

**Question.** In R1b every crop arm raised false alarms 1.7–2.9×, and even the full
frame names activities that aren't there. Can a second, cheap step remove the false
ones without losing the real ones, and at what token cost? Backlog #17.

### Facts, measured before registering (R1b's first-pass answers, no VLM)

| first pass | fired windows naming ≥ 1 letter | letters named | correct | precision | top false letter |
|---|---|---|---|---|---|
| 4B F | 732 of 1,531 (48%) | 1,383 | 391 | 0.28 | G sits/stands: 640 of 992 |
| 4B GO | 1,137 (74%) | 3,038 | 907 | 0.30 | G: 1,070 of 2,131 |
| 8B F | 1,229 (80%) | 2,074 | 680 | 0.33 | B vehicle starts/stops: 638 of 1,394 |
| 8B GO | 1,040 (68%) | 1,689 | 736 | 0.44 | D object: 338 of 953 |

### Arms

The first pass is R1b's, reused unchanged (same windows, frames, model and greedy
decoding). Server settings as R1b (prefix caching off).

| arm | what it adds |
|---|---|
| Fc, GOc | **stricter prompt**, one pass: F's or GO's request with one added line ("Most moments from this camera show none of these. Name a letter only if you can clearly see it happening; if unsure, leave it out."). Replaces the first pass. |
| FV, GOV | **verification**: for each fired window whose first answer named letters, one more request with the **same images**, asking yes/no for each named letter only. A letter survives if the answer is yes. Unparsed letters are kept (and counted). |
| F4V8 | **model cascade**: the 4B's F answers verified by the **8B**. The 8B runs only on windows the 4B flagged. |

**Models:** 4B: Fc, GOc, FV, GOV. 8B: Fc, FV, F4V8.

### Scoring

As R1b: lift above chance per activity instance, false alarms per hour, paired
bootstrap over clips. Added: **precision** (correct letters / letters named, over
fired windows), **keep rates** for correct and false letters (V arms), and **tokens
per fired window** summed over both passes (F4V8 reports 4B and 8B tokens
separately).

### Predictions

- **R1c.1** Verification halves the 4B's false alarms: FV's false alarms per hour
  **≤ 0.5×** F's, with lift **≥ F's − 2 points**.
- **R1c.2** Verification makes crops affordable: GOV's false alarms per hour **≤ 4B
  F's**, with lift **≥ 4B F's + 5 points**.
- **R1c.3** A stricter prompt helps less: Fc cuts false alarms **≥ 25%** against F,
  at **≤ 5%** more tokens, but **by less than** FV does.
- **R1c.4** Self-checking is not blind agreement: in FV (4B), false letters survive
  at **≤ half** the rate of correct letters, and **≥ 80%** of correct letters
  survive.
- **R1c.5** The 8B verifies itself too: 8B FV's false alarms per hour **≤ 0.5×** 8B
  F's, lift within **2 points**.
- **R1c.6** The cascade is the cheap route to the 8B: F4V8's lift is **within 5
  points** of 8B F's, at **≤ 0.5×** its false alarms per hour, while running the 8B
  on **≤ 50%** of fired windows.

**Decision rule, stated in advance:** if R1c.1 or R1c.2 holds, the recipe adds
"verify before alerting" as the default for alerting use; if R1c.6 holds, the
4B-screens-8B-confirms cascade becomes the recommended configuration. If none
holds, precision needs a trained head (R4).

### Not measured

Prefix caching (the verify request repeats the images, so with caching on most of
its prefill would be a cache hit; R1c runs with caching off, as R1b did, and
reports tokens); live cameras and cameras per GPU; verification of M, GS or GSC;
per-letter thresholds.

## R2 · Manufacturing: crops where people are large — pre-registered 2026-09-25, before any measurement

**Question.** R1 and R1b tested crops on surveillance video, where people are small
and most actions are body-scale. Theory (Context-Aware RCNN, ViCrop) says crops pay
when the answer lies in **detail the full frame shrinks away**. HA4M is that case:
a worker at a fixed station assembles a planetary gear set in 12 steps, and most
steps differ only by **which small white part is in the hand**. Do label-free hand
crops beat the full frame here?

### Data: HA4M (CC BY 4.0; Cicirelli et al., Scientific Data 2022)

- 217 recordings, 41 workers, one fixed Azure Kinect in front of the bench,
  2048 × 1536 colour frames, every frame labelled with its step (0 = idle, 1–12).
  Only the frames used were downloaded (15.6 GB of a 4.6 TB share,
  `tools/fetch_ha4m.py`); **the data is deleted from the rig when R2 is done.**
- **Two camera setups**, read from the image (nearest reference frame): a lab where
  parts sit in clear boxes (22 workers) and a white room where they lie loose and
  the worker stands further away (19 workers). One worker per setup is held out
  (IDU001, IDU023): their first recording supplies the **reference sheet**, and none
  of their samples are scored.
- **Samples:** two frames 1 s apart around the midpoint of every step segment, and
  of the final idle stretch where it is ≥ 31 frames. **2,570 scored** (1,133 lab,
  1,437 white room).

### Facts, measured before registering (`experiments/r2_prep.py`, no VLM)

- YOLOv8s-pose finds the worker on all 5,844 sampled frames. A hand crop falls back
  to the full frame for 31 samples (Kinect) and 22 (YOLO).
- Hand crops cover **~2.8% of the frame** at native resolution, so parts appear
  **2.6× larger** than in the full frame (which is shrunk to 38%). The Kinect and
  YOLO crops agree at a median IoU of 0.78.

### Arms

Every request carries the setup's **reference sheet** first (12 numbered pictures,
one per step, from the held-out worker), then two frames one second apart, unless
marked "names only". Frames are capped at 451,584 px, context frames at 112,896 px.
The model answers with a step number (1–12) or 0.

| arm | what the VLM sees after the sheet |
|---|---|
| F | the full frame |
| B | a fixed bench ROI per setup (5th–95th percentile box of all Kinect wrist positions, ×1.2), native |
| HK | a hand crop from the **Kinect's body tracking**: one shoulder width square around each wrist, union over hands and frames, native |
| HY | the same rule from **YOLOv8s-pose** (RGB only) |
| HKC | HK plus the full frame at low resolution |
| F-n, HK-n | F and HK with the step names only, no sheet |

**Models:** Qwen3-VL-8B on every arm; Qwen3-VL-4B on F, HK and HKC.

### Scoring

Accuracy is predicting the step exactly. Chance comes from shuffling each arm's
answers across its samples (20 shuffles), and lift is accuracy minus chance. Arms
are compared by a **paired bootstrap over workers** (39 scored). Breakdowns: part
steps (1–8, 10, 11) against the whole-assembly steps (9, 12); lab against white
room.

### Predictions (Qwen3-VL-8B unless stated)

- **R2.1** Hand crops beat the full frame: HK's lift ≥ F's **+ 5 points**.
- **R2.2** An RGB-only hand finder suffices: HY within **5 points** of HK.
- **R2.3** HK's gain over F is **larger on part steps** (1–8, 10, 11) than on 9 and 12.
- **R2.4** HK's gain over F is **larger in the white room** (smaller parts in frame)
  than in the lab.
- **R2.5** A fixed bench ROI gains **less** than hand crops: B − F < HK − F.
- **R2.6** Context does not hurt: **HKC ≥ HK**.
- **R2.7** The reference sheet matters: F ≥ F-n **+ 10** and HK ≥ HK-n **+ 10**.
- **R2.8** HK costs **≤ 70%** of F's prompt tokens (the sheet's cost is shared).
- **R2.9** The smaller model gains more: Qwen3-VL-4B's HK − F ≥ the 8B's.

### Not measured

Crops inside DeepStream; live cameras; frames beyond two; a trained action
recogniser on HA4M (R4); depth.

## R2b · Does the crop hold more information for a trained head? — pre-registered 2026-09-27, before any measurement

**Question.** In R2, zero-shot Qwen3-VL-8B did worse with hand crops, but it was at
the floor (14.5% on 12 steps) and could not name the parts at any resolution.
That leaves two explanations. **(A)** The crop holds more usable detail that a model
without knowledge of the parts cannot use; if so, training on the domain pays and
crops help. **(B)** The crop loses what identifies the step (where the hand goes,
how far the assembly has got); if so, crops are the wrong tool even for a trained
model. A probe on the frozen encoder separates the two cheaply, before any
fine-tuning is spent.

### Design

- **Data:** HA4M, a 20-worker subset re-downloaded for this probe (10 lab workers,
  IDU001–010; 10 white-room workers, IDU023–032), with the same sampling as R2: two
  frames 1 s apart around each step's midpoint. **Deleted again when R2b is done.**
- **Encoder:** Qwen3-VL-8B's vision tower, frozen, loaded alone. Each image's final
  merged tokens are mean-pooled into one vector, and the two frames are averaged.
- **Views**, all from the Kinect's body tracking (label-free), images preprocessed as
  the VLM would (frames capped at 451,584 px):
  - **full**: the full frame
  - **tight**: R2's hand crop, a square of one shoulder width around each wrist
  - **wide**: the same at **three** shoulder widths (hands, upper body, workspace)
  - **bench**: the per-setup station ROI, as in R2
  - **full + wide**: the two vectors concatenated
- **Head:** multinomial logistic regression on standardised features, with L2 strength
  chosen by an inner split of the training workers. **5-fold cross-validation grouped
  by worker**, so every prediction is for a worker the head never saw. Metrics:
  accuracy and macro-F1 on steps 0–12. Views are compared by a paired bootstrap over
  workers.

### Predictions

- **R2b.1** The encoder carries the step: the full-frame probe reaches **≥ 40%**
  accuracy (zero-shot, R2: 14.5%).
- **R2b.2** Crops help a trained head: **wide ≥ full + 3 points**.
- **R2b.3** Context matters: **wide ≥ tight**.
- **R2b.4** Detail and context are complementary: **full + wide ≥ the better single
  view + 2 points**.
- **R2b.5** Wide's gain over full is **larger on part steps** (1–8, 10, 11) than on
  9 and 12.

**Decision rule, stated in advance:** fine-tune a sub-1B VLM on crops only if R2b.2
or R2b.4 holds with a 95% interval above zero. Otherwise the crop thread closes.

### Not measured

Fine-tuning; temporal models; the YOLO hand finder (R2: within 0.7 points of the
Kinect's); other encoders.
