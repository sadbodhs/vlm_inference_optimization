# VLM Inference Optimization

Measured answers to: **what does a vision-language model actually cost to serve, and
what does making it cheaper cost you in accuracy?**

Single RTX 3090, 24 GB, sm_86. Sibling study to
[computer_vision_optimization](https://github.com/sadbodhs/computer_vision_optimization),
which asked the same kind of question about YOLO serving stacks.

Status: **harness built and dry-run verified. No GPU numbers yet.**

---

## Why this is a different problem from CV serving

The YOLO study had one bottleneck axis and constant quality. A VLM breaks that:

1. **Two engines with opposite needs.** A vision encoder (compute-bound, fixed cost)
   feeding an autoregressive LLM (prefill compute-bound, decode *bandwidth*-bound).
2. **Input size is a free variable.** An image is not one frame, it is *N tokens*, and
   N is a policy choice — resolution, tiling, pruning. Nothing in the CV study had
   this knob.
3. **Quality is a dependent variable.** Every token-reduction and quantization trick
   trades accuracy, so the deliverable is a **latency × accuracy frontier**, not a
   ranking.

## The experiments

| | Question | Status |
|---|---|---|
| **E0** saturation | At what arrival rate does this arm stop keeping up — and was the *client* the real bottleneck? | harness verified |
| **E1** roofline | Does single-stream decode land where 936 GB/s says it must? | harness verified |
| **E2** TTFT vs vision tokens | What does one vision token cost, and what is the fixed floor underneath it? | harness verified |
| **E3** token budget | How far can the budget fall before accuracy does — and does the limit depend on the task? | harness verified |

E0 runs first on purpose. It is the only experiment that can invalidate all the others.

## The 3090 sets the questions

Decode reads every weight once per token, so at 936 GB/s the ceiling is fixed before
any software choice:

| Config | Weights | Decode ceiling | KV room at 0.9 util |
|---|---|---|---|
| 7B FP16 | ~16.5 GB | ~55 tok/s | ~5 GB (~90 K tokens) |
| 7B AWQ | ~6 GB | ~155 tok/s | ~15 GB (~270 K tokens) |
| 3B FP16 | ~7.5 GB | ~125 tok/s | ~14 GB (~380 K tokens) |

At ~1,500 vision tokens for a document page that is **~60 concurrent requests at FP16
versus ~175 at AWQ**. On this card quantization is primarily a *batching*
optimization, not a latency one. E0 and E1 exist to test that claim.

No FP8 on sm_86 — that arm is out of scope here rather than faked.

## Run it

Everything runs in Docker — harness included. No host Python, no venv, no conda.

```bash
docker build -f docker/harness.Dockerfile -t vlmbench-harness .
./scripts/smoke.sh        # all four experiments against a mock server, no GPU
```

Against a real server on the 3090:

```bash
docker/run_vllm.sh start arms/B_vllm_awq.yaml       # first run downloads weights
docker/run_harness.sh python3 experiments/e0_saturation.py     --arm arms/B_vllm_awq.yaml
docker/run_harness.sh python3 experiments/e1_roofline.py       --arm arms/B_vllm_awq.yaml
docker/run_harness.sh python3 experiments/e2_ttft_vs_tokens.py --arm arms/B_vllm_awq.yaml
docker/run_harness.sh python3 experiments/e3_token_budget.py   --arm arms/B_vllm_awq.yaml \
    --manifest data/docvqa/manifest.jsonl --scorer anls
docker/run_harness.sh python3 experiments/plot.py
docker/run_vllm.sh stop
```

Server and harness sit on a shared `vlmbench` Docker network and address each other
by container name, so the measurement code is byte-identical across arms (R1). The
resolved image digest of both is written into every `meta.json`.

The harness container is CPU-only and requests `--gpus all` for exactly one reason:
the NVIDIA runtime injects `nvidia-smi`, without which every run silently records
null clocks and the thermal check stops working.

## How the harness holds the bar

The [four rules](https://sadbodhs.github.io/) are enforced in code, not in prose.

- **R1 — one variable at a time.** Every stack is driven through one
  OpenAI-compatible streaming path. An arm is a YAML file; adding one never adds a
  branch.
- **R2 — saturate before you measure.** Every run self-reports `saturated`,
  `not-saturated`, or `client-bound`. The third is the one nobody checks: if the load
  generator falls behind schedule, the throughput number describes the harness. It is
  detected by tracking per-request scheduling lag, and such runs are discarded rather
  than reported. Saturation itself is judged on *queue growth* — Poisson arrival
  variance at small n is ±18%, enough to fake a saturation signal on its own.
- **R3 — publish the cost.** Accuracy is scored from the same generated outputs as
  the latency run, so a frontier point is never assembled from two experiments.
- **R4 — say what was not measured.** Every run writes `not_measured` into
  `meta.json`.

Thermal state is recorded before and after every run, since a 350 W card throttling
across a long sweep produces drift that reads exactly like an effect.

## Layout

```
bench/          harness: client, load generators, metrics, scorers
experiments/    E0-E3 + plotting
arms/           one YAML per configuration; arms are data, not code
tools/          mock VLM server for GPU-free dry runs
data/           frozen eval subsets + manifests
results/        raw jsonl, summaries, meta
docs/           write-ups
PLAN.md         full project plan
```

## Not measured yet

Everything. No number in this repo came off a GPU. Specifically open: whether CPU-side
image preprocessing outweighs the vision encoder in TTFT (E2); whether prefix caching
changes the stack ranking under realistic multi-turn reuse; whether a disaggregated
encoder service pays for its embedding-transport hop; video, multi-image, and anything
requiring FP8.
