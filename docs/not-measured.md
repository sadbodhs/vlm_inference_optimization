# Not measured

What these numbers do **not** entitle you to claim.

## Scope

- **Two checkpoints, one family.** Qwen2.5-VL-7B-AWQ and Qwen3-VL-8B-AWQ. No
  InternVL, no FP16 comparison — so nothing here separates "how VLM serving behaves" from "how
  this checkpoint behaves on this card".
- **Appearance measured on one dataset, capped by it.** VQAv2 is the only
  text-free task, and its COCO images are small, so its curve stops at 385 tokens
  (native size). High-resolution appearance tasks — small distant objects, fine
  surveillance detail — are untested; the feature-size rule predicts they behave
  like reading, not like VQAv2. See [the per-task frontier](task-frontier.md).
- **Short outputs only.** Every accuracy number comes from ~6-token answers, so
  everything measured is prefill-dominated. Decode at 88.7% of roofline is
  currently a rounding error against 700 ms of prefill; on a reasoning workload it
  would become the bottleneck, and these conclusions may not transfer.

## Statistical resolution

- **One run per point** for E0 and E4 — no error bars on those. E1 and E2 carry
  three repeats (spread 0.11–0.14%).
- **n=500 for accuracy**, which resolves differences above roughly 0.03 ANLS.
  Anything smaller is unmeasured, not flat.

## Known-unresolved

- **The SGLang accuracy gap.** 3–5 ANLS points, cause unproven. The `triton_attn`
  vision-attention hypothesis is untested.
- **E2 under-predicts real documents by 6–18%.** Synthetic images of equal token
  count are cheaper than real ones; the cause is unexplained.

## Out of scope on this rig

- **No FP8** — sm_86 has no FP8 tensor cores. Excluded rather than faked.
- **E4 claims no accuracy.** The clip is 3D animation.
- **E4 decode cost is unattributed** — decoders run in separate containers.
- **One frame rate per stream.** Motion-gated and keyframe sampling, which is what
  a real deployment would use, are unmeasured.
- **Prefix caching under realistic reuse.** Measured at 0% and ~100% reuse; the
  interesting middle is unmeasured.
