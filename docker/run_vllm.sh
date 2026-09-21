#!/usr/bin/env bash
# vLLM server for a given arm, in its own container on the shared bench network.
#   docker/run_vllm.sh start arms/B_vllm_awq.yaml
#   docker/run_vllm.sh stop
#
# VLLM_IMAGE defaults to :latest for convenience. Pin it to a DIGEST before any
# run whose numbers you intend to publish -- the resolved digest is recorded into
# every meta.json either way, so a `latest` run is traceable but not repeatable.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
NET="${BENCH_NET:-vlmbench}"
NAME="${VLLM_NAME:-vlm-server}"
PORT="${VLLM_PORT:-8000}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

case "${1:-start}" in
  start)
    ARM="${2:?usage: $0 start <arm.yaml>}"
    MODEL=$(grep -E '^model:' "$ARM" | head -1 | sed 's/^model:[[:space:]]*//')
    QUANT=$(grep -E '^quantization:' "$ARM" | head -1 | sed 's/^quantization:[[:space:]]*//')

    docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    mkdir -p "$HF_CACHE"

    ARGS=(--model "$MODEL" --port "$PORT"
          --max-model-len "${MAX_MODEL_LEN:-8192}"
          --gpu-memory-utilization "${GPU_UTIL:-0.90}"
          --limit-mm-per-prompt '{"image":1}')
    [ -n "$QUANT" ] && [ "$QUANT" != "null" ] && ARGS+=(--quantization "$QUANT")

    echo "starting $NAME  image=$IMAGE  model=$MODEL"
    docker run -d --rm --name "$NAME" --network "$NET" \
      --gpus all --ipc=host --shm-size=8g \
      -v "$HF_CACHE:/root/.cache/huggingface" \
      -p "127.0.0.1:$PORT:$PORT" \
      ${HF_TOKEN:+-e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"} \
      "$IMAGE" "${ARGS[@]}" >/dev/null

    echo "waiting for /v1/models (first run downloads weights; watch: docker logs -f $NAME)"
    for _ in $(seq 1 "${WAIT_S:-1800}"); do
      curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "ready"; exit 0; }
      docker ps -q -f "name=$NAME" | grep -q . || { echo "container exited:"; docker logs --tail 30 "$NAME"; exit 1; }
      sleep 1
    done
    echo "timed out" >&2; exit 1
    ;;
  stop) docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" ;;
  digest) docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}' ;;
  *) echo "usage: $0 [start <arm.yaml>|stop|digest]" >&2; exit 2 ;;
esac
