# Literature log

Prior work that bears on Track B, with the numbers that matter to us. Checked
2026-09-24; add entries as they are read, not as they are cited.

## Cropping around actors (trained models)

**Context-Aware RCNN** — Wu et al., ECCV 2020. [arXiv 2007.09861](https://arxiv.org/abs/2007.09861)
Crop each actor, resize, classify its action (AVA v2.1, 3D ResNet-50).
- crop + resize vs RoI pooling from the full frame: **25.0 vs 22.1 mAP**
- gain by actor box size, extra-small → extra-large: **+4.0, +3.0, +1.8, +0.7, −0.2**
- input resolution 224 / 192 / 160 / 112: 25.0 / 24.3 / 23.7 / 21.4
- box expansion 1.2 / 1.5 / 1.8 / 2.0 / 2.5×: **24.1 / 25.0 / 24.8 / 25.0 / 24.3**
- context-dependent classes lose: "watch (TV)" −6.8 AP, "listen to" −6.1
- *For us:* the reference-level prediction (1.5–2×) and the size prediction
  (small actors gain most) come from here. It is trained on crops; we are zero-shot.

**Open-Vocabulary Spatio-Temporal Action Detection** — Wu et al., 2024. [arXiv 2405.10832](https://arxiv.org/abs/2405.10832)
Fine-tunes a video-language model on region-text pairs; beats zero-shot VLM
baselines on UCF-JHMDB and OV-AVA. *For us:* the trained alternative R4 should cite.

## Cropping and zooming for VLMs (zero-shot)

**ViCrop — MLLMs Know Where to Look** — Zhang et al., ICLR 2025. [arXiv 2502.17422](https://arxiv.org/abs/2502.17422)
Crops chosen from the model's own attention, given **in addition to** the full image.
TextVQA (LLaVA-1.5) 47.8 → 56.1; V* 42.4 → 62.3; no loss on large-object benchmarks.
*For us:* arm 3 (crop + full frame).

**GapSight — Learning to Look Again** — 2026. [arXiv 2608.21762](https://arxiv.org/abs/2608.21762)
A learned router decides when to crop; InternVL2.5-8B six-benchmark average 52.3 →
64.3, above CropVLM, ViCrop and ZoomRefine. *For us:* cropping is task-selective.

**ZoomEye** — EMNLP 2025. [ACL Anthology](https://aclanthology.org/2025.emnlp-main.335/)
Tree search over zoomed sub-regions; InternVL2.5-8B +15.7 / +17.7 on HR-Bench.

**CropVLM** — 2025. [arXiv 2511.19820](https://arxiv.org/abs/2511.19820)
A trained cropping model for fine-grained perception.

## Marking instead of cropping

**What does CLIP know about a red circle?** — Shtedritski et al., ICCV 2023. [arXiv 2304.06712](https://arxiv.org/abs/2304.06712)
A red circle on the full image keeps context; keypoint naming on CUB 46.5% vs 25.5%
for cropping. *For us:* arm 2 (marked full frame).

## Scale

**SNIP** — Singh & Davis, CVPR 2018. [arXiv 1711.08189](https://arxiv.org/abs/1711.08189)
Detectors are not scale-invariant; train and test at matched object scales.
*For us:* why a best person size in pixels (a "reference level") should exist.

## VLMs on surveillance and industry

**Evaluation of Vision-LLMs in Surveillance Video** — 2025. [arXiv 2510.23190](https://arxiv.org/abs/2510.23190)
Gemma-3-4B, NVILA-8B, Qwen2.5-VL-7B, VideoLLaMA-3-7B on UCF-Crime and RWF-2000;
82–86% on RWF-2000; anonymisation costs 2–11 points. No cropping or cost study —
the gap Track B fills.

**MMAD** — ICLR 2025. [arXiv 2410.09453](https://arxiv.org/abs/2410.09453)
MLLMs on industrial anomaly detection (8,366 images): GPT-4o 74.9%, "far short of
industrial requirements"; humans +4, experts +12. *For us:* defect inspection is
out of scope for the recipe; human activity and state are in.

## Datasets to evaluate

- **HA4M** — 41 subjects, 12 assembly actions, RGB / depth / skeleton. [Scientific Data 2022](https://www.nature.com/articles/s41597-022-01843-z)
- **InHARD** — industrial human action recognition, > 2M frames, 16 subjects.
- **AVA v2.x** — per-person atomic actions with boxes; the standard for R5.
