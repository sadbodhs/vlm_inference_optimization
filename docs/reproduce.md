# Reproduce

Everything runs in Docker — harness included. No host Python, no venv, no conda.

## Without a GPU

```bash
docker build -f docker/harness.Dockerfile -t vlmbench-harness .
./scripts/smoke.sh
```

Runs all experiments against a mock server that has the qualitative shape of a VLM
(vision tokens scale with pixels, everything slows under load, there is a finite
batch). It exists so harness bugs surface locally instead of on the rig at 2am.

## Against a real server

```bash
scripts/fetch_model.sh arms/B0_vllm_awq_clean.yaml
N=1000 scripts/fetch_dataset.sh docvqa

docker/run_server.sh start arms/B0_vllm_awq_clean.yaml
scripts/run_experiments.sh arms/B0_vllm_awq_clean.yaml
docker/run_server.sh stop
```

Swap `arms/C0_sglang_awq_clean.yaml` to run the same experiments on SGLang — the
launcher dispatches on the arm's `stack:` field, so nothing else changes.

## Live video

```bash
docker/run_decoders.sh start 8
docker/run_harness.sh python3 experiments/e4_video_stream.py \
    --arm arms/B0_vllm_awq_clean.yaml --streams 8 --duration 45
docker/run_decoders.sh stop
```

## The rig

RTX 3090, 24 GB, sm_86 — **no FP8**, so that arm is out of scope rather than faked.
Under sustained load the card is **power-capped** (349 W of 350 W) at 66 °C, not
thermally throttled; the distinction comes from the driver's own clock-event
reasons, because a clock-ratio test calls a healthy 3090 throttled.
