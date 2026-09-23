#!/usr/bin/env bash
# Run a detector command in the CV study's image (ultralytics + torch + cv2).
#   docker/run_yolo.sh python3 experiments/e7_detect.py --sizes 640,1280
#
# Same rule as run_harness.sh, in both directions: a detector timing run and a
# VLM measurement sharing the GPU would each absorb the other's load, and nothing
# in either output would say so.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${YOLO_IMAGE:-vlmbench-yolo:latest}"   # docker/Dockerfile.yolo
MODELS="${MODELS_DIR:-$HOME/sadbodh/model_exports}"

if [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
  # vlm-server claims 90% of the card; a detector beside it risks CUDA OOM on
  # either side and contaminates both timings. The co-hosted run is a separate,
  # deliberate experiment (ALLOW_CONCURRENT=1), never an accident.
  BUSY=$(docker ps --format '{{.Names}}' | grep -E '^(harness-|yolo-|vlm-server$)' || true)
  if [ -n "$BUSY" ]; then
    echo "REFUSING TO START: another measurement container is running:" >&2
    echo "$BUSY" | sed 's/^/  /' >&2
    exit 1
  fi
fi

exec docker run --rm -i --gpus all \
  --name "yolo-$$" \
  --ipc host \
  -v "$PWD:/work" -w /work \
  -v "$MODELS:/models:ro" \
  -e PYTHONPATH=/work \
  -e YOLO_CONFIG_DIR=/tmp/ultralytics \
  -e GIT_SHA="$(git rev-parse HEAD 2>/dev/null || true)" \
  -u "$(id -u):$(id -g)" \
  --entrypoint "" \
  "$IMAGE" "$@"
