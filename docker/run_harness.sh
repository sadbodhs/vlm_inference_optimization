#!/usr/bin/env bash
# Run any harness command inside the harness container.
#   docker/run_harness.sh python3 experiments/e1_roofline.py --arm arms/B_vllm_awq.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${HARNESS_IMAGE:-vlmbench-harness:latest}"
NET="${BENCH_NET:-vlmbench}"

docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null

# The harness does no CUDA work. It asks for the GPU only so the NVIDIA container
# runtime injects nvidia-smi -- without it, every run records null temperature and
# clocks, and the thermal-drift check quietly stops working.
# Refuse to run two experiments against one server at once. A load test and a
# latency probe sharing a server silently corrupt each other: the latency numbers
# absorb the load test's queueing, and nothing in the output says so. Concurrency
# is a variable the arm controls, never an accident of what else was running.
if [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
  RUNNING=$(docker ps -q --filter "ancestor=$IMAGE" | wc -l | tr -d ' ')
  if [ "$RUNNING" != "0" ]; then
    echo "REFUSING TO START: $RUNNING harness container(s) already running." >&2
    docker ps --filter "ancestor=$IMAGE" --format '  {{.Names}}  {{.RunningFor}}' >&2
    echo "Two experiments against one server contaminate each other." >&2
    echo "Wait, or set ALLOW_CONCURRENT=1 if you genuinely mean to overlap them." >&2
    exit 1
  fi
  # ...and never alongside a detector run, which shares the GPU (docker/run_yolo.sh)
  YOLO_BUSY=$(docker ps --format '{{.Names}}' | grep -E '^yolo-' || true)
  if [ -n "$YOLO_BUSY" ]; then
    echo "REFUSING TO START: a detector run is using the GPU: $YOLO_BUSY" >&2
    exit 1
  fi
fi

GPU_ARGS=()
if [ "${BENCH_GPU:-auto}" != "off" ] && docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then
  GPU_ARGS=(--gpus all)
fi

exec docker run --rm -i \
  "${GPU_ARGS[@]}" \
  --name "harness-$$" \
  --network "$NET" \
  -v "$PWD:/work" \
  $([ -d "${FRAME_DIR:-/tmp/vlm_frames}" ] && echo "-v ${FRAME_DIR:-/tmp/vlm_frames}:${FRAME_DIR:-/tmp/vlm_frames}:ro") \
  -w /work \
  -e HARNESS_IMAGE="$(docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo "$IMAGE")" \
  -e SERVER_IMAGE="${SERVER_IMAGE:-$(docker image inspect "${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)}" \
  -e PYTHONPATH=/work \
  -e GIT_SHA="$(git rev-parse HEAD 2>/dev/null || true)" \
  -e GIT_DIRTY="$(git diff --quiet 2>/dev/null && echo clean || echo dirty)" \
  -u "$(id -u):$(id -g)" \
  "$IMAGE" "$@"
