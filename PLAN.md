# VLM Inference Optimization — Project Plan

Sibling study to [`computer_vision_optmization`](https://github.com/sadbodhs/computer_vision_optmization).
Same method (controlled arms, one harness, measured not asserted), different physics.

---

## 1. Why this is not just "the YOLO study with a bigger model"

The previous study had one bottleneck axis: how fast can frames get through a fixed-cost
forward pass. Transport and batching were the whole story, and quality was constant —
FP16 vs INT8 YOLO is a small mAP delta you can report once and forget.

A VLM breaks that in three ways:

1. **Two engines, not one.** A ViT-ish vision encoder (compute-bound, fixed cost, exactly
   like YOLO — *your existing skills apply directly here*) bolted to an autoregressive LLM
   (prefill compute-bound, decode memory-bandwidth-bound). They want opposite optimizations.
2. **The input size is a free variable.** An image is not 1 frame, it is *N tokens*, and N is
   a policy decision (resolution, tiling, pixel-shuffle, pruning). Prefill cost is roughly
   quadratic-ish in N at long context and linear in the encoder. Nothing in the YOLO study
   had this knob.
3. **Quality is now a dependent variable.** Every token-reduction and quantization trick
   trades accuracy. So the deliverable is not "which stack is fastest" — it is a
   **latency × accuracy Pareto frontier**, per task type.

That third point is the thesis of the repo. The headline result should be something like:
*"For document VQA you cannot cut vision tokens below X without falling off a cliff; for
natural-image VQA you can cut 60% for free. Here is the frontier, here is where each
serving stack lands on it."* That is a genuinely useful artifact and nobody publishes it
cleanly.

---

## 2. Metrics (define these on day 1, never change them)

| Metric | Definition | Why it matters |
|---|---|---|
| **TTFT** | request sent → first output token | Dominated by preprocess + encoder + prefill. The VLM-specific metric. |
| **TPOT / ITL** | mean inter-token latency after the first | Pure decode; memory-bandwidth bound |
| **E2E** | request → last token | What a user feels |
| **Throughput** | completed req/s, and output tok/s | Server economics |
| **Goodput @ SLO** | req/s meeting e.g. TTFT<1s AND TPOT<50ms | The only throughput number that is honest |
| **Peak VRAM** | steady-state, under load | Decides what fits on the 3090 |
| **Accuracy** | task metric (ANLS, exact-match, relaxed-acc) | The axis the YOLO study didn't have |

Decompose TTFT explicitly into: `image I/O+decode | CPU preprocess | encoder | projector | LLM prefill`.
**I expect CPU-side image preprocessing (PIL resize/normalize) and vision-token count to be
larger TTFT contributors than people assume, and the vision encoder itself to be smaller.**
Proving or killing that hypothesis is Phase 2 and it is the most citable thing here.

---

## 3. Hardware: RTX 3090 (confirmed) — and what it dictates

**RTX 3090: 24 GB GDDR6X, ~936 GB/s bandwidth, Ampere sm_86.** Three consequences that
decide model selection before taste does.

**(a) No FP8.** FP8 tensor cores need Ada/Hopper. Your quantization arms are FP16/BF16
(Ampere does support BF16), INT8, and W4A16 (AWQ / GPTQ, with Marlin kernels — these run
well on Ampere). FP8 weights and FP8 KV cache are simply off the table. Say so in the README
rather than quietly omitting it; if you later rent an L40S or H100 for a week, FP8 becomes a
clean bolt-on chapter rather than a retrofit.

**(b) Decode is bandwidth-bound, and 936 GB/s sets a hard ceiling you can predict today.**
Single-stream decode reads the full weight set once per token, so:

```
max decode tok/s  ≈  936 GB/s ÷ (weight bytes in VRAM)
```

| Config | Weights in VRAM | Roofline decode ceiling |
|---|---|---|
| 7B-class VLM, FP16 | ~16.5 GB | **~55 tok/s** |
| 7B-class VLM, AWQ W4A16 | ~6 GB | **~155 tok/s** |
| 3B-class VLM, FP16 | ~7.5 GB | **~125 tok/s** |

This is the single most useful number on the page. It predicts that **4-bit quantization
roughly triples single-stream decode on this card**, and it gives you a week-1 sanity check:
if your harness reports TPOT implying a rate meaningfully *above* the roofline, the harness
is wrong (usually double-counting a warmup or missing a sync). Measure it and put the
measured-vs-roofline ratio in the docs — it is the kind of thing that makes a benchmark repo
trustworthy.

**(c) 24 GB is the real constraint, and it is a weights-vs-KV tradeoff.** Rough budget at
`gpu_memory_utilization=0.9` (~21.5 GB usable):

| Config | Weights | Left for KV + activations | ≈ KV tokens* |
|---|---|---|---|
| 7B FP16 | ~16.5 GB | ~5 GB | ~90 K |
| **7B AWQ (ViT kept FP16)** | **~6 GB** | **~15 GB** | **~270 K** |
| 3B FP16 | ~7.5 GB | ~14 GB | ~380 K |

\* assuming a Qwen2.5-7B-style LLM: 28 layers × 4 KV heads × 128 head-dim × 2 (K,V) × 2 bytes
≈ **56 KB per token**. Compute this from the actual `config.json` for whatever you pick —
treat my numbers as the method, not the answer.

Put that in workload terms: a DocVQA page at ~1,500 vision tokens means **7B FP16 gives you
roughly 60 concurrent requests of KV, while 7B AWQ gives you ~175**. On a 24 GB card,
quantization is not primarily a latency optimization — **it is a batching optimization**, and
that reframing is worth a paragraph in the write-up.

### Model shortlist for this card

- **Primary / spine: Qwen2.5-VL-7B-Instruct, AWQ 4-bit, vision tower left FP16.**
  Best fit for 24 GB by the table above; native dynamic resolution makes the Phase 5
  token-budget sweep a continuous knob instead of a handful of discrete tiling modes; widest
  support across vLLM / SGLang / TRT-LLM. Quantizing the ViT usually costs more accuracy than
  it saves memory (it is <1.5 GB) — verify once, then leave it FP16.
- **Control: Qwen2.5-VL-3B-Instruct, FP16.** Same architecture, different capacity. This pair
  is what lets you say "that difference is model *size*, not model *design*" — and it runs
  un-starved, so it separates real effects from 3090 memory pressure.
- **Second family (only if doing the cross-section): InternVL 2.5 8B, AWQ.** Tiling +
  pixel-shuffle instead of dynamic resolution — the check that your conclusions are about
  VLMs and not about Qwen's preprocessor.
- **Fixed-resolution control (cheap, optional): a SigLIP-based fixed-token model.** Constant
  vision-token count isolates LLM-side effects from vision-token effects.

**Excluded by the hardware, and say why in the README:** anything ≥30B even at 4-bit (~19 GB
of weights leaves no usable KV cache — it will "run" and benchmark terribly, which is a
hardware artifact, not a finding); 11B+ VLMs at FP16; and any FP8 checkpoint.

### The max_pixels trap — set this before your first run

Dynamic-resolution models default to very generous pixel caps, and a full-page DocVQA scan
can expand into many thousands of vision tokens. On a 3090 that produces a single request
whose prefill stalls every other request in flight, and your early numbers will be
incomprehensible. **Cap `max_pixels` explicitly in every arm config and log the resulting
vision-token count per request** (the harness already records it, per §2). The cap is not a
nuisance parameter — it *is* the Phase 5 independent variable, so treat it as first-class
from day one.

### Thermals will corrupt a long sweep

A 3090 is a 350 W card that throttles under sustained load. Across hundreds of arms run
back-to-back, late arms run slower than early ones and you will mistake drift for effect.
Mitigations, all cheap: lock clocks (`nvidia-smi -lgc`), log GPU temperature and SM clock
into every `meta.json`, insert a fixed cooldown between arms, and **randomize arm execution
order** so any residual drift becomes noise rather than a systematic bias toward whichever
arm you happened to run first.

**Datasets** (keep them small and fixed — 200-500 samples each, checked into `data/`):
- **DocVQA** — dense text, high-resolution, token-hungry. The "hard" end.
- **ChartQA** — structured reasoning, medium resolution.
- **TextVQA** or **VQAv2** — natural images, low token need. The "easy" end.
- **MMMU** subset — optional, for reasoning-heavy long-output behavior.

The spread between DocVQA and VQAv2 *is* the result. Don't drop one.

---

## 4. The model axis — multiple models, without a combinatorial explosion

**First, a gate: do you need this?** Multi-model is *optional depth*, not a requirement.
Skip it and stay single-model if the goal is a mechanism study — Phases 2, 3 and 5 are
complete and publishable with one model. Add models only when you want to claim
**generalization** ("this optimization transfers across VLM architectures") or when the
comparison itself is the product ("which VLM is cheapest per unit of accuracy"). The
structure below is designed so you can start single-model and bolt models on later without
redoing anything — the spine runs first and stands alone.

Running several VLMs is the right call, and it upgrades the repo's strongest claim from
*"this optimization worked"* to *"this optimization generalizes"*. But it is also the fastest
way to kill the project: 6 models × 50 arms × 3 datasets × load profiles is not a study, it
is a compute bill. Structure it as a **spine plus cross-sections**.

**Spine (deep):** one primary model — Qwen2.5-VL-7B-AWQ — runs *every* arm in every phase.
All the mechanism findings come from here.

**Cross-section (wide):** all models run a *fixed, small* set of ~6 arms chosen to test
whether spine findings transfer. Suggested cross-section: `{baseline FP16, AWQ, best token-budget
setting, prefix-caching on, chunked-prefill on, arm E disaggregated}` × 3 datasets. That is
tractable and answers the generalization question.

### Why different VLMs are not interchangeable here

Families differ most in **how many vision tokens they emit per image**, which is the single
biggest determinant of serving cost — and it varies by close to an order of magnitude at the
same input resolution. That variation is under-reported and measuring it cleanly is a
contribution on its own.

| Family | Vision-token strategy | What it tests |
|---|---|---|
| **Qwen2.5-VL** (7B + 3B) | Native dynamic resolution, window attention in the ViT | Token count scales with image size — the token-budget knob is continuous |
| **InternVL** | Tiled AnyRes + pixel-shuffle compression | Tiling policy as the knob; very different prefill shape |
| **SmolVLM / small class** | Aggressive pixel shuffle, few tokens by design | The "already optimized" control — do your tricks have anything left to win? |
| **Gemma 3 / PaliGemma-style** | Fixed-resolution SigLIP, constant token count | The constant-cost control: isolates LLM-side effects from vision-token effects |
| **Pixtral / Molmo** (optional) | Another dynamic-resolution take | Guards against "conclusion is really about Qwen" |

Pick **3-4**, not all of them — and on a 24 GB card the shortlist in §3 already picks them
for you: **Qwen2.5-VL-7B-AWQ** (spine), **Qwen2.5-VL-3B FP16** (size-vs-design control),
**InternVL 8B AWQ** (different tiling), plus optionally a fixed-resolution model. The 3B/7B
pair is doing real work: without it you cannot tell whether a difference is architecture or
just parameter count. Everything ≥30B is excluded by VRAM, not by preference.

### Compare at iso-accuracy, not iso-settings

Default settings give different models wildly different token budgets, so a raw
latency table is close to meaningless — it mostly reports who chose a smaller default
resolution. Report at least one of:

- **Iso-accuracy latency:** tune each model's token budget until it hits the same DocVQA
  score, then compare TTFT and throughput. This is the number a practitioner actually wants.
- **Accuracy per GPU-second** (or per unit of goodput), plotted as a frontier with all
  models on the same axes.

The headline plot becomes one Pareto chart per dataset, with a curve per model rather than a
single point per model. A 3B model that lands above the 7B curve on natural images and far
below it on documents is exactly the kind of result worth publishing.

### Two things multiple models unlock

1. **Transfer matrix.** Rows = optimizations, columns = models, cells = speedup and accuracy
   delta. An optimization that helps one family and hurts another is a finding, not a bug —
   e.g. attention-based token pruning should behave differently on a model whose ViT already
   compresses via pixel shuffle.
2. **Cascade / routing arm (Phase 6, arm G).** Small model answers first; escalate to the
   large model on low confidence or on a document-type trigger. Measure end-to-end goodput
   and accuracy versus always-large. On 24 GB this is feasible if both are AWQ (~2 GB + ~5 GB
   weights), which leaves room for KV cache — one of the few genuinely multi-model serving
   optimizations you can run on a single 3090.

### Cost control

Multi-model doubles infrastructure risk, so: one Docker image per *stack*, models mounted as
volumes not baked in; model identity is a field in the arm YAML, never a code branch; and
every model gets a **smoke arm** (10 samples) that must pass before it joins a full sweep.

---

## 5. Phases

### Phase 0 — Harness first (days 1-3)
Do not run a single benchmark until the harness exists. In the last project the harness
was implicit; here it must be the product, because you will run hundreds of arms.

Build `bench/` with one contract:

```
run(arm_config, dataset, load_profile) -> results/<arm_id>/{latency.jsonl, accuracy.json, meta.json}
```

- `latency.jsonl`: one row per request — TTFT, ITL list, E2E, in/out token counts, vision token count.
- `meta.json`: git SHA, GPU, driver, container digest, every knob — plus **GPU temp and SM
  clock at start and end of the run** (see the thermals note in §3). Reproducibility is a
  documented strength of the previous repo — keep it.
- Load generator with **Poisson arrivals at a target rate**, not a fixed concurrency loop.
  Closed-loop concurrency hides queueing and makes every stack look better than it is.
- Accuracy scorer runs on the *same* generated outputs as the latency run, not a separate
  pass. Otherwise you can't put a point on the Pareto plot.

**Gate: you can produce one plot from one command before moving on.**

### Phase 1 — Baselines (days 3-6)
Three arms, out-of-box, no tuning:
- **A. HF `transformers`** naive generate, batch 1 — the deliberately slow reference.
- **B. vLLM** default settings.
- **C. SGLang** default settings.

Purpose: establish the span and validate the harness against three different servers.
Expect the span to be large (A is often an order of magnitude off). Record it; the rest of
the project is about explaining the gap.

### Phase 2 — Decompose TTFT (week 2) ⭐ the interesting chapter
The analog of your stage-decomposition chapter, and where the novel content is.
- Instrument with Nsight Systems + torch profiler; NVTX ranges around preprocess / encoder /
  projector / prefill.
- Sweep image resolution and tiling and plot **vision tokens → TTFT**. Find the knee.
- Separate CPU preprocessing from GPU work explicitly. Then kill the CPU part with
  **nvImageCodec / DALI / NVJPEG** GPU decode+resize and re-measure.
- Report a stacked-bar TTFT breakdown per dataset. DocVQA and VQAv2 will look completely
  different and that picture is the repo's thumbnail.

### Phase 3 — Vision path optimization (week 3) — your home turf
This is where the TensorRT/Triton experience transfers 1:1.
- Export the vision tower to ONNX → TensorRT. FP16, then INT8 with calibration.
- Compare against `torch.compile` and against the stack's built-in encoder path.
- Batch the encoder across requests (multi-image and multi-tile requests are free batching
  opportunities that most stacks under-exploit).
- **Cache image embeddings** keyed by image hash — for multi-turn chat over one document
  this removes the encoder entirely from turns 2..N.
- Watch for the trap: a 3× faster encoder may move E2E by 5% if the encoder was never the
  bottleneck. Report that honestly — it's a better finding than a fake win.

### Phase 4 — Token budget & the Pareto frontier (week 4) ⭐ the thesis
- Arms: resolution caps, tile-count caps, pixel-unshuffle ratios, and training-free token
  pruning/merging (FastV-style attention-based pruning, VisionZip-style selection, simple
  spatial pooling as the dumb baseline).
- For each arm measure TTFT **and** accuracy on all three datasets.
- Plot accuracy vs TTFT per dataset. Identify the cliff per task type.
- Hypothesis to test: *the dumb baseline (spatial pooling) is competitive with clever
  pruning on natural images and much worse on documents.* If true, that's a strong,
  practical, publishable claim.

### Phase 5 — LLM-side serving (week 5)
- Quantization arms: FP16 / AWQ W4A16 / GPTQ, vision tower left FP16 (quantizing it usually
  hurts more than it helps — verify, don't assume).
- **KV cache**: INT8 KV, cache size sweeps, and the resulting max-concurrency effect.
- **Prefix caching** — the biggest practical VLM win and under-benchmarked. Same image,
  many questions; long shared system prompt. SGLang's RadixAttention vs vLLM's APC. Design
  a workload with realistic reuse (e.g. 5 turns per document) — *and* one with zero reuse,
  so you report the win honestly rather than at its best case.
- **Chunked prefill** — matters enormously here because a document image is a huge prefill
  that stalls every decode in flight. Measure the TTFT/TPOT tradeoff curve.
- CUDA graphs, `max_num_seqs` / `max_num_batched_tokens` sweeps.

### Phase 6 — Architecture arms (weeks 6-7) — the "six pipelines" analog
| ID | Arm | Question it answers |
|---|---|---|
| A | HF transformers | reference floor |
| B | vLLM monolithic | the default choice |
| C | SGLang | does RadixAttention change the ranking under reuse? |
| D | TensorRT-LLM multimodal (± Triton frontend) | does the compiled stack still win when a ViT is in the path? |
| E | **Disaggregated encoder**: vision tower as its own TensorRT/Triton service, embeddings shipped to a vLLM LLM service | **your unique contribution** |
| F | Prefill/decode disaggregation (stretch) | if you get multi-GPU or rented hardware |
| G | Small→large cascade (only if running multi-model) | can routing beat always-large on goodput at equal accuracy? |

**Arm E is the one to lean into.** It is the direct descendant of your CUDA-IPC / shared-memory
transport work, it lets the encoder and the LLM scale and quantize independently, and the
interesting cost is the embedding transport — hundreds of KB to MBs per request. Measure
gRPC vs CUDA IPC vs shared memory for that hop. You already know how to do this and nobody
else in the VLM space is measuring it carefully.

### Phase 7 — Load & serving behavior (week 8)
- Concurrency/arrival-rate sweeps → goodput-vs-SLO curves per arm.
- Mixed workload: short natural-image queries interleaved with huge document requests.
  Head-of-line blocking on the big prefills is the thing to expose.
- Tail latency (p95/p99 TTFT), not just means.

### Phase 8 — Video VLM (stretch, week 9) — the bridge back to DeepStream
Frames → VLM is where your two repos meet: NVDEC decode, frame-sampling policy (uniform vs
keyframe vs motion-gated), token budget per frame, streaming summarization. Even a small
chapter here makes the two repos read as one body of work.

### Phase 9 — Write-up (week 9-10)
Mirror the existing structure: `README.md` (results-first), `STORY.md` (narrative),
mkdocs site, `docs/` chapters per phase, `results/` raw data, CI that at minimum lints and
rebuilds the docs. Lead the README with the Pareto plot, not the architecture diagram.

---

## 6. Repo skeleton

```
vlm_inference_optimization/
├── .github/workflows/       # docs build, lint, smoke test
├── bench/                   # harness: load gen, metrics, scorers, plotting
│   ├── harness.py
│   ├── loadgen.py           # Poisson arrivals
│   ├── metrics.py           # TTFT/ITL/goodput
│   └── scorers/             # ANLS, relaxed accuracy, EM
├── arms/                    # one config per arm; arms are data, not code
│   ├── A_hf_baseline.yaml
│   ├── B_vllm_default.yaml
│   └── ...
├── serving/
│   ├── vllm/  sglang/  trtllm/  triton/
│   └── encoder_service/     # Phase 6 arm E
├── vision/                  # ONNX export, TRT build, GPU preprocessing
├── data/                    # fixed eval subsets + manifest
├── docker/                  # one image per stack, digests pinned
├── results/                 # raw jsonl, committed
├── docs/                    # mkdocs chapters
├── scripts/
├── README.md  STORY.md  mkdocs.yml
```

Keep **arms as YAML config, never as branching code**. You will have 50+ arms; if each one
is an `if` statement the repo dies in week 4.

---

## 7. Start here (first 3 days, concretely)

1. `git init vlm_inference_optimization`, drop in the skeleton and this plan as `docs/plan.md`.
2. Pin the environment: one Dockerfile for vLLM with an exact image digest. Record driver,
   CUDA, GPU in `meta.json` from the start.
3. Download and freeze the eval subsets (200 samples each, DocVQA / ChartQA / VQAv2),
   commit the manifest with checksums.
4. Write `bench/harness.py` against **one** backend (vLLM's OpenAI-compatible server —
   streaming responses give you TTFT and ITL for free) and one arm.
5. Produce the first plot: TTFT vs vision-token-count on DocVQA, batch 1. One command.
6. Only then add arms.

---

## 8. Risks

- **24 GB ceiling.** Every batching/KV result is 3090-flavored — see the weights-vs-KV table
  in §3. State it up front, use the 3B control to show which conclusions survive, and report
  KV-tokens-available alongside every throughput number so readers can rescale to their own card.
- **Thermal drift across long sweeps.** Lock clocks, log temps, randomize arm order (§3).
  This is the most likely source of a confidently-wrong result in the whole project.
- **Stack churn.** vLLM/SGLang/TRT-LLM multimodal support changes fast. Pin container
  digests and record versions in every `meta.json` or your results become unreproducible
  within months.
- **Accuracy scoring drift.** Use the official metric per dataset (ANLS for DocVQA, relaxed
  accuracy for ChartQA) and freeze the scorer early — a mid-project scorer change invalidates
  every earlier Pareto point.
- **Scope.** Phases 2, 4 and arm E are the original contributions. If time runs short, cut
  Phase 8 and half of Phase 6, not those.

---

## 9. Naming note

Your existing repo is `computer_vision_optmization` (missing the `i` in "optimization") and
this local folder repeats it. Worth deciding now whether the new repo matches the typo for
consistency or spells it correctly — mixed spelling across two linked repos is the worst of
the three options.
