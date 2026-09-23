#!/usr/bin/env bash
# Start the serving stack an arm asks for. One entry point, so an arm changes the
# stack by changing a YAML field rather than by calling a different script -- which
# is what keeps R1 ("one variable at a time") true by construction instead of by
# discipline.
#
#   docker/run_server.sh start arms/C_sglang_awq.yaml
#   docker/run_server.sh stop
set -euo pipefail
cd "$(dirname "$0")/.."

NET="${BENCH_NET:-vlmbench}"
NAME="${SERVER_NAME:-vlm-server}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# grep exits 1 on a missing optional field; pipefail then trips set -e and kills a
# multi-hour run with no error in the log. See the arm_field commit.
arm_field () {
  grep -E "^$2:" "$1" 2>/dev/null | head -1 | sed "s/^$2:[[:space:]]*//" || true
}

preflight_vram () {
  local util="$1"
  read -r FREE TOTAL < <(nvidia-smi --query-gpu=memory.free,memory.total \
      --format=csv,noheader,nounits | tr -d ',')
  local want
  want=$(python3 -c "print(int($TOTAL * $util))")
  if [ "$want" -gt "$FREE" ]; then
    echo "REFUSING TO START: utilisation $util asks for ${want} MiB of ${TOTAL} MiB," >&2
    echo "but only ${FREE} MiB is free. Holders:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2
    exit 1
  fi
  echo "  vram ok: want ${want} MiB, free ${FREE} MiB of ${TOTAL} MiB"
}

wait_ready () {
  local port="$1" secs="${2:-1800}"
  for _ in $(seq 1 "$secs"); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { echo "ready"; return 0; }
    docker ps -q -f "name=$NAME" | grep -q . || {
      echo "container exited:" >&2; docker logs --tail 40 "$NAME" >&2; return 1; }
    sleep 1
  done
  echo "timed out waiting on :$port" >&2; return 1
}

case "${1:-start}" in
  start)
    ARM="${2:?usage: $0 start <arm.yaml>}"
    STACK=$(arm_field "$ARM" stack)
    MODEL=$(arm_field "$ARM" model)
    QUANT=$(arm_field "$ARM" quantization)
    REV=$(arm_field "$ARM" revision)
    EXTRA=$(arm_field "$ARM" server_args)
    # A video prompt carries several frames; the default of one image per prompt
    # rejects them outright.
    MAXIMG=$(arm_field "$ARM" max_images)
    MAXIMG="${MAXIMG:-1}"
    MAXLEN=$(arm_field "$ARM" max_model_len)
    MAXLEN="${MAXLEN:-${MAX_MODEL_LEN:-8192}}"
    UTIL="${GPU_UTIL:-0.90}"
    [ -n "$MODEL" ] || { echo "no 'model:' in $ARM" >&2; exit 1; }

    docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    mkdir -p "$HF_CACHE"
    # The VRAM preflight alone does not catch a detector: YOLO holds ~0.6 GB, so
    # 90% still "fits" -- and then both grow into the same 24 GB. Refuse outright.
    if [ "${ALLOW_CONCURRENT:-0}" != "1" ] && docker ps --format '{{.Names}}' | grep -qE '^yolo-'; then
      echo "REFUSING TO START: a detector run (docker/run_yolo.sh) is using the GPU." >&2
      exit 1
    fi
    preflight_vram "$UTIL"

    case "$STACK" in
      vllm)
        IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}"
        PORT="${SERVER_PORT:-8000}"
        ARGS=(--model "$MODEL" --port "$PORT"
              --max-model-len "$MAXLEN"
              --gpu-memory-utilization "$UTIL"
              --limit-mm-per-prompt "{\"image\":$MAXIMG}")
        [ -n "$QUANT" ] && [ "$QUANT" != "null" ] && ARGS+=(--quantization "$QUANT")
        [ -n "$REV" ] && ARGS+=(--revision "$REV")
        ENTRY=()
        ;;
      sglang)
        IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.20}"
        PORT="${SERVER_PORT:-30000}"
        # mem-fraction-static is NOT the analogue of vLLM's gpu-memory-utilization,
        # despite both being "a fraction of the card".
        #
        #   vLLM  gpu-memory-utilization : TOTAL budget, profiled, activations inside
        #   SGLang mem-fraction-static   : the STATIC pool only, activations on top
        #
        # Passing 0.90 to both gave SGLang a 23.01 GiB static pool of a 23.56 GiB
        # card. Small images were fine; the vision encoder then OOMed on full-size
        # documents and 475 of 500 requests returned HTTP 500. It also inflated its
        # KV pool to 261k tokens against vLLM's 213k, which reads like an engine
        # advantage and is really just a bigger budget.
        #
        # So the arm names this explicitly rather than inheriting GPU_UTIL.
        MEMFRAC=$(arm_field "$ARM" mem_fraction_static)
        MEMFRAC="${MEMFRAC:-0.79}"
        ARGS=(python3 -m sglang.launch_server --model-path "$MODEL"
              --host 0.0.0.0 --port "$PORT"
              --context-length "$MAXLEN"
              --mem-fraction-static "$MEMFRAC")
        [ -n "$QUANT" ] && [ "$QUANT" != "null" ] && ARGS+=(--quantization "$QUANT")
        [ -n "$REV" ] && ARGS+=(--revision "$REV")
        ENTRY=(--entrypoint "")
        ;;
      *) echo "unknown stack '$STACK' in $ARM (want vllm|sglang)" >&2; exit 2 ;;
    esac
    # shellcheck disable=SC2206
    [ -n "$EXTRA" ] && ARGS+=($EXTRA)

    echo "starting $NAME  stack=$STACK  image=$IMAGE  port=$PORT"
    docker run -d --rm --name "$NAME" --network "$NET" \
      --gpus all --ipc=host --shm-size=16g \
      -v "$HF_CACHE:/root/.cache/huggingface" \
      -p "127.0.0.1:$PORT:$PORT" \
      ${HF_TOKEN:+-e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"} \
      "${ENTRY[@]}" "$IMAGE" "${ARGS[@]}" >/dev/null

    wait_ready "$PORT" "${WAIT_S:-1800}"
    ;;
  stop) docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "not running" ;;
  digest)
    ARM="${2:?usage: $0 digest <arm.yaml>}"
    case "$(arm_field "$ARM" stack)" in
      vllm)   docker image inspect "${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}" --format '{{index .RepoDigests 0}}' ;;
      sglang) docker image inspect "${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.20}" --format '{{index .RepoDigests 0}}' ;;
    esac
    ;;
  *) echo "usage: $0 [start <arm.yaml>|stop|digest <arm.yaml>]" >&2; exit 2 ;;
esac
