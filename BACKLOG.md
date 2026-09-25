# Backlog

Every idea lands here first. Status: **idea → planned → pre-registered → running →
published / dropped**. Pre-registrations live in `PLAN.md` (Track A) or
`RECIPE.md` (Track B).

| # | idea | from | hypothesis | cost | track | status |
|---|---|---|---|---|---|---|
| 1 | actor-crop reference level (margin 1.2–3×, person 112/224/448 px, native) | E7d crop results; Context-Aware RCNN | an optimum margin of 1.5–2×; gains largest for small actors | ~1 GPU day | B · R1 | published (R1) |
| 2 | marked full frame (box drawn around the actor) | red-circle paper | beats the plain full frame at zero token cost | small, inside R1 | B · R1 | published (R1) |
| 3 | crop *plus* a small full frame (ViCrop-style) | ViCrop | recovers context-dependent actions the crop loses | small, inside R1 | B · R1 | published (R1) |
| 4 | ideal crop from ground-truth boxes vs tracker boxes | tracker ceiling 59% / 31% | separates "the idea works" from tracker misses | small, inside R1 | B · R1 | published (R1) |
| 5 | per-actor scoring, prompt adapted to crops | E7 scoring was per window | fair test of person-level actions | code only | B · R1 | published (R1) |
| 6 | group crops: several images in one request vs separate requests | user, 2026-09-24 | one request is cheaper per crop under load | small | B · R1 | idea |
| 7 | crop + JPEG encode inside DeepStream (GPU) instead of disk JPEGs | live runs read pre-extracted JPEGs | realistic cost; cameras may drop slightly | ~1 day eng. | B | idea |
| 8 | manufacturing domain (HA4M or InHARD) | user, 2026-09-24 | cropping helps more when people are large in frame | data + ~1 GPU day | B · R2 | idea — licences to check |
| 9 | whole-scene domain (traffic) | recipe must say when not to crop | full frame wins; the gate still pays | data + ~1 GPU day | B · R3 | idea |
| 10 | trained action detector baseline | reviewers will ask | trained wins on closed classes, VLM on open ones | ~1 week | B · R4 | idea |
| 11 | AVA subset + second model family (InternVL / Gemma) | generality | ordering of crop policies holds across families | ~1 week | B · R5 | idea |
| 12 | more frames of the crop instead of more pixels | E6 (frames fix ordering) | 6–8 crop frames beat 2 full frames at ≤ equal tokens | ~0.5 GPU day | B · R6 | idea |
| 13 | detector ceiling at ~18–20 cameras (interval, smaller detector) | E7d: det p99 3 s at 19–20 cams | lighter detection lifts the ceiling | ~0.5 GPU day | A/B | idea |
| 14 | MPS for the detector + VLM on one card | E7b/E7c "not measured" | less interference, +1–2 cameras | ~0.5 GPU day | A | idea |
| 15 | Qwen3.5 prose answers with a longer answer budget | E7d | recovers the 2B at many times the decode cost | small | A | idea |
| 16 | R1b: crops chosen without labels (tracker proximity groups), negatives from gate-fired windows, crop + context as lead arm | R1: oracle boxes leak participants; crops name an activity for 86–100% of uninvolved people | the single-person +12–16 point gain survives label-free crops; crop + context keeps false alarms near the full frame's | ~0.5 GPU day | B · R1b | pre-registered |
