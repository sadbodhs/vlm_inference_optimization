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
GPU_ARGS=()
if [ "${BENCH_GPU:-auto}" != "off" ] && docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then
  GPU_ARGS=(--gpus all)
fi

exec docker run --rm -i \
  "${GPU_ARGS[@]}" \
  --name "harness-$$" \
  --network "$NET" \
  -v "$PWD:/work" \
  -w /work \
  -e HARNESS_IMAGE="$(docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo "$IMAGE")" \
  -e SERVER_IMAGE="${SERVER_IMAGE:-}" \
  -e PYTHONPATH=/work \
  -u "$(id -u):$(id -g)" \
  "$IMAGE" "$@"
