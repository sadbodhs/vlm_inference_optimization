#!/usr/bin/env bash
# Run a command in the E7c DeepStream image (docker/Dockerfile.ds).
#   docker/run_ds.sh python3 experiments/e7c.py offline
# Same GPU guard as the other launchers; the live run sets ALLOW_CONCURRENT=1
# because sharing the card with vlm-server IS that experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${DS_IMAGE:-vlmbench-ds:latest}"
NET="${BENCH_NET:-vlmbench}"
if [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
  BUSY=$(docker ps --format '{{.Names}}' | grep -E '^(harness-|yolo-|ds-|vlm-server$)' || true)
  if [ -n "$BUSY" ]; then
    echo "REFUSING TO START: GPU in use by: $BUSY" >&2; exit 1
  fi
fi
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
NAME="ds-$$"
# DS_TIMEOUT=<seconds>: kill the CONTAINER when it expires. `timeout` around this
# script only kills the docker client; the container kept a hung pipeline -- and
# 2.6 GB of GPU memory -- for 32 minutes after its timeout "fired".
if [ -n "${DS_TIMEOUT:-}" ]; then
  ( sleep "$DS_TIMEOUT"; docker kill "$NAME" >/dev/null 2>&1 && echo "WATCHDOG: killed $NAME after ${DS_TIMEOUT}s" >&2 ) &
fi
exec docker run --rm -i --gpus all --name "$NAME" --network "$NET" --ipc host \
  -v "$PWD:/work" -w /work \
  -v "$HOME/sadbodh/model_exports/e7c:/models" \
  -v "$HOME/sadbodh/deepstream-yolov11:/opt/ds:ro" \
  -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1 \
  -e GIT_SHA="$(git rev-parse HEAD 2>/dev/null || true)" \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  --entrypoint "" \
  "$IMAGE" "$@"
