#!/usr/bin/env bash
# R2b (RECIPE.md): frozen-encoder probe on HA4M. Feature extraction needs the GPU
# and runs in the vLLM image (torch + transformers); prep and probe need neither.
#   scripts/r2b.sh prep
#   scripts/gpu_lock.sh run r2b -- scripts/r2b.sh features [--limit N]
#   scripts/r2b.sh probe
# When R2b is published, delete data/ha4m again (the user asked).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?prep|features|probe}"; shift || true
case "$MODE" in
  prep|probe)
    docker/run_harness.sh python3 experiments/r2b_probe.py "$MODE" "$@" ;;
  features)
    docker run --rm --gpus all --ipc=host --name "r2b-$$" \
      -v "$PWD:/work" -w /work -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e PYTHONPATH=/work/experiments:/work -e PYTHONUNBUFFERED=1 \
      --entrypoint python3 "${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}" \
      experiments/r2b_probe.py features "$@" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
