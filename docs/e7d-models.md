# E7d · Which VLM behind the gate: size and generation

**Question:** after [E7c](e7c-deepstream.md) the VLM is the only limit on how many
cameras one RTX 3090 carries. Is a smaller VLM the lever? Does a *newer* small model
keep the recognition an older, larger one had? And is E7's finding that the VLM
"cannot see phones or conversations" a property of the task, or of the model?

**Answer:**

- **Generation beats size.** **Qwen3-VL-8B** recognises activities at **+39.7 points
  above chance** against Qwen2.5-VL-7B's +16.9 (the only statistically clear
  difference in the set) **and** carries one more camera (10 vs 9). It is better than
  the 7B on both axes.
- **E7's perception limit was the model.** Qwen3-VL-8B recognises conversations at
  **+34 points** above chance, Qwen3.5-9B at +28; the 7B was *below* chance. Phones
  stay hard for every model (≤ +12).
- **Newer 4B models keep the 7B's recognition at 1.7–1.8× the cameras:**
  Qwen3-VL-4B **16** full-frame, Qwen3.5-4B **18** with ROI crops (7B: 9 and 10).
  Qwen2.5-VL-3B matches the 7B at 14 and 15.
- **At 2B the detector becomes the limit, at 18 cameras.** Both 2B models hit it
  with VLM capacity to spare, but their answers are unusable: one ticks every box,
  the other answers in prose.
- **A small model frees memory only if you ask it to.** At the same util 0.80 every
  model peaked at 21.6–23.5 GiB of 24. Given half the budget (0.40), Qwen3-VL-2B kept
  its 18 cameras in **13 GiB**, and Qwen3-VL-4B its 16 in **12 GiB** once its context
  length was sized to the ~1,100-token requests.

Pre-registered in [PLAN.md §13](https://github.com/sadbodhs/vlm_inference_optimization/blob/main/PLAN.md)
before any measurement.

## Setup

- **Eight 4-bit models, one variable.** Everything but the model is E7/E7c's:
  vLLM v0.29.0, 16 images per request, 16,384 context, no prefix caching, util
  0.80, temperature 0, 16 output tokens, the same prompt and the same pixels
  (≤ 451,584 per frame). Checkpoints are pinned by revision in `arms/E_*.yaml`.

    | generation | models | image tokens per call (full / ROI) |
    |---|---|---|
    | Qwen2.5-VL (Jan 2025) | 7B, 3B | 1,298 / 1,056 |
    | Qwen3-VL (Oct 2025) | 8B, 4B, 2B | 1,035 / 842 |
    | Qwen3.5 (Feb 2026) | 9B, 4B, 2B — thinking off | 1,041 / 848 |

    The newer generations use 32 px per visual token against 28, so the same pixels
    cost them ~20% fewer tokens. Pixels are equalised; tokens are not.
- **Recognition:** E7's VLM stage on every window of all 24 clips (3,600 full-frame,
  2,192 ROI), scored against the shuffled-answer chance baseline. Models are
  compared by a **paired bootstrap over clips**: each draw resamples clips once and
  scores both models on it.
- **Cameras:** E7c's live pipeline unchanged (DeepStream + NvDCF tracker gate,
  vLLM on the same card), stepping camera counts up by 2 until two failures, then
  filling the gap. Same bar: ≥ 95% of answers under 2 s old, detection p99 < 1 s.

## Recognition

![Recognition above chance by activity group, eight models](img/e7d-groups.png)

| model | on disk | above chance, full frame | vs 7B (95% interval) | false alarms / h | answers in format |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B (E7) | 6.9 GB | +16.9 | — | 608 | 100% |
| Qwen2.5-VL-3B | 3.4 GB | +16.9 | +0.0 [−10.8, +8.4] | 674 | 100% |
| **Qwen3-VL-8B** | 7.6 GB | **+39.7** | **+22.8 [+8.5, +31.8]** | 804 | 100% |
| Qwen3-VL-4B | 4.4 GB | +21.6 | +4.7 [−7.0, +13.9] | 726 | 100% |
| Qwen3-VL-2B | 2.2 GB | +7.0 | −9.9 [−35.8, +4.1] | 9,254 | 100% |
| Qwen3.5-9B | 9.1 GB | +28.5 | +11.6 [−10.3, +24.6] | 1,512 | 96% |
| Qwen3.5-4B | 4.0 GB | +19.5 | +2.6 [−15.7, +11.7] | **102** | 95% |
| Qwen3.5-2B | 2.5 GB | 0 | −16.9 [−40.2, −0.8] | 0 | **0%** |

With ROI crops: Qwen3-VL-8B +36.6 (+26.7 over the 7B, [+10.6, +43.4]), Qwen3.5-9B
+25.8, Qwen3.5-4B +24.7, Qwen3-VL-4B +23.3, Qwen2.5-VL-3B +16.7, Qwen2.5-VL-7B +9.9.

**What the table says:**

- **Only Qwen3-VL-8B is clearly better** than the 7B. Every ≤ 4B model is
  statistically indistinguishable from it; two hours of video from 24 cameras
  cannot separate differences smaller than ~10 points.
- **Conversations are visible to newer models.** Qwen2.5-VL-7B: −9 on
  "people talk to or touch each other". Qwen3-VL-8B: **+34**. Qwen3.5-9B: +28.
  Qwen3.5-4B: +16. E7 read the 7B's failure as the
  [feature-size rule](task-frontier.md), a limit of the task at surveillance
  distance. It was a limit of that model.
- **Phones remain hard for all of them:** the best is +12, and n = 50.
- **Qwen3.5-4B is the precise one:** one-sixth of the 7B's false alarms, at the same
  recognition. It answers "N" (nothing happening) on 82% of windows.
- **Same score, different strengths.** The 3B beats the 7B on getting into or out of
  vehicles (+39 vs +24) and loses on sitting or standing (+27 vs +74, n = 10).

### Two failure modes at 2B

- **Qwen3-VL-2B asserts nearly everything.** Its most common answer is
  "A, B, C, D, E, F, G, H" (926 of 3,600 windows). Its chance baseline is 90%, so
  almost nothing it says carries information.
- **Qwen3.5-2B does not follow the format.** With thinking off it still reasons in
  prose: *"To determine the correct answer, we must analyze…"*. The 16-token
  answer budget ends before any letter. A longer budget might recover an answer, at
  many times the decode cost; that was not the pre-registered test.

The larger Qwen3.5 models do this too, though less: with ROI crops, 25% (4B) and
33% (9B) of their answers are prose, scored as asserting nothing. Their ROI numbers
are a lower bound.

## Cameras per 3090

![Recognition against cameras per 3090](img/e7d-frontier.png)

| model | full frame, `track-motion` | ROI, `person-track-motion` | what failed one camera later |
|---|---|---|---|
| Qwen2.5-VL-7B ([E7c](e7c-deepstream.md)) | 9 | 10 | the VLM |
| Qwen2.5-VL-3B | 14 | 15 | the VLM |
| **Qwen3-VL-8B** | **10** ¹ | **11** ¹ | the VLM |
| **Qwen3-VL-4B** | **16** | 17 | the VLM (+ detector at ROI) |
| Qwen3-VL-2B | 18 | 18 | **the detector** |
| Qwen3.5-9B | 10 | 11 | the VLM |
| **Qwen3.5-4B** | 15 | **18** | the VLM (+ detector at ROI) |
| Qwen3.5-2B | 18 | 18 | **the detector** |

¹ From a rerun with one discarded warm-up run. The first sweep read **5** full-frame
and **10** ROI: right after the server started, 6 and 8 cameras failed on detection
latency (1.8 s and 1.2 s p99) while the VLM's answers at 8 were 100% fresh, and the
ROI sweep lost its summary to a teardown hang (below). Both sweeps are in
`results/e7d/E_q3vl_8b/`.

**Cameras follow the vision encoder and the token count, not the parameter count.**
The 3B gains 1.5× on a 2.3× smaller language model because it keeps the 7B's
~670M-parameter vision encoder. The newer generations spend ~20% fewer tokens on
the same pixels (32 px per token against 28), part of why their 4B models carry
more cameras than Qwen2.5's 3B.

### The detector ceiling at 18 cameras

Every sweep that reached 18–20 cameras (both 2B models, and both 4B models with
ROI crops) failed there on **detection** latency: DeepStream's p99 passed 1 s
(1.3–3.5 s) whether or not the VLM kept up. At 19 cameras the 2B models' answers
were still 89–99.7% fresh. With a batch-16 engine detecting every 6th frame, the CV
side runs out of time beside the VLM at about 18 cameras on this card. More cameras
from here need lighter detection or a second GPU, not a smaller VLM.

## Memory: the budget, not the model, decides

vLLM sizes itself from `--gpu-memory-utilization`, not from the model: a smaller
model spends what it saves on KV cache. Under load every model then grew past its
0.80 budget (19.2 GiB), as in [E7b](e7b-live.md). No allocation failed.

| at util 0.80 | GPU memory at the supported limit | highest seen in any run |
|---|---|---|
| Qwen2.5-VL-7B (E7c) | 20.6–22.5 GiB | 22.7 GiB |
| Qwen2.5-VL-3B | 21.2–23.2 GiB | 23.4 GiB |
| Qwen3-VL-8B | 21.0–22.0 GiB | 22.2 GiB |
| Qwen3-VL-4B | 22.2–23.0 GiB | 23.4 GiB |
| Qwen3-VL-2B | 22.3–22.5 GiB | **23.5 GiB** |
| Qwen3.5-9B / 4B / 2B | 20.8–22.0 GiB | 22.3 GiB |

The pre-registered test: the ≤ 4B model with the most full-frame cameras, rerun at
**util 0.40**. The rule picked Qwen3-VL-2B (18 cameras):

| Qwen3-VL-2B | util 0.80 | **util 0.40** |
|---|---|---|
| cameras, full frame / ROI | 18 / 18 | **18 / 18** |
| GPU memory at the limit | 22.5 / 22.3 GiB | **12.4 / 13.1 GiB** |
| highest seen | 23.5 GiB | **13.4 GiB** |

**Ten gigabytes freed, no camera lost.** Because the rule's pick gives unusable
answers, the best usable small model, Qwen3-VL-4B, was run too, and **it did not
start at 0.40**. One request at the registered 16,384-token context needs 2.25 GiB
of KV cache, and 2.01 GiB was left after weights and profiling. Qwen3-VL-4B stores
144 KB of KV per token, 2.6× Qwen2.5's. E7d's requests use ~1,100 tokens, so it was
rerun with a 4,096-token context, a deviation from the registration:

| Qwen3-VL-4B | util 0.80, 16,384 context | **util 0.40, 4,096 context** |
|---|---|---|
| cameras, full frame / ROI | 16 / 17 | **16 / 18** |
| GPU memory at the limit | 22.2 / 23.0 GiB | **12.1 / 12.4 GiB** |
| highest seen | 23.4 GiB | **12.7 GiB** |

**The usable 4B model keeps its cameras in about half the card.** The ROI result
is one camera higher at 0.40, within the ±1 of a single run. So a small model does
free the card, but only once both the memory budget and the context length are sized
to the requests actually sent. Left at the defaults, vLLM claims the memory anyway.

## Predictions: which held

| # | prediction (pre-registered) | measured | verdict |
|---|---|---|---|
| 12 | Qwen2.5-VL-3B carries 1.4–2.0× the 7B's cameras | 1.56× full, 1.5× ROI | **held** |
| 13 | …and its full-frame lift is lower than the 7B's | +0.0 points [−10.8, +8.4] | **failed** — no measurable loss |
| 14 | a ≤ 4B Qwen3-VL / Qwen3.5 model is not significantly below the 7B **and** carries ≥ 1.4× its cameras | Qwen3-VL-4B: +4.7 [−7.0, +13.9], 1.78×; Qwen3.5-4B: +2.6 [−15.7, +11.7], 1.67× | **held** |
| 15 | no model beats chance by > 10 points on phones or conversations | Qwen3-VL-8B +34 on conversations, +12 on phones | **failed** |
| 16 | at util 0.40 the chosen model loses ≤ 1 camera | Qwen3-VL-2B (the rule's pick): 18 → 18; Qwen3-VL-4B: 16 → 16, with a 4,096 context | **held** — the 4B only after the context deviation |

## Deviations from the registration

- **Qwen3-VL-8B's cameras come from a rerun** with one discarded warm-up run (both
  sweeps reported above).
- **The memory test ran a second model**, because the rule's pick gives unusable
  answers, and that model needed a **4,096-token context** to start at 0.40.
- **Teardown bounded mid-experiment.** At 11 cameras one of eleven sources never
  delivered end-of-stream, and the run waited 75 minutes for the watchdog. From
  Qwen3.5-9B's ROI sweep on, teardown gives up after 60 s. That is after the
  measurement window, so it cannot change a number, and the summary is now saved
  after every camera count. It fired three more times.

## Caveats

- **Recognition intervals are wide.** 24 clips, and activities within a clip are
  not independent. Only the Qwen3-VL-8B difference clears it.
- **One prompt, one answer format, 16 tokens.** A model that reasons first is
  penalised by design: a deployment parsing letters would get nothing either.
- **One 120 s run per camera count;** limits are ±1 camera. The first run after a
  server start can fail on detection latency alone (the 3B at 6 cameras, 1.3 s p99;
  the 8B above); the 3B's limit is unaffected because 8–14 all passed.
- **Memory carries over between runs.** vLLM keeps what it grows, so later runs
  start higher. Peaks are per run, not per camera count in isolation.
- **Live frames are pre-extracted JPEGs** read from disk, as in E7b and E7c; a
  deployment would crop and encode from the decoder's GPU buffers (Track B backlog).
- **Not measured:** models outside the Qwen family, FP16 variants, thinking mode,
  Qwen3.5-0.8B, Qwen3.6 (27B and 35B only), MPS.

**Next:** [Track B](recipe.md) takes the crop question further: an actor-centred
crop at a reference margin and size, rather than one box around everyone.
