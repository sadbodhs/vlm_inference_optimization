#!/usr/bin/env bash
# Does vLLM v0.30.0 fix either EVS failure seen in E8 on v0.29.0?
#   1. Qwen3-VL + EVS: every video request crashed the engine (evs.py:338).
#   2. Qwen2.5-VL-7B + EVS 0.75: vision-encoder OOM with ~23 requests in flight.
# Exploratory, not an experiment. Under the GPU lock:
#   scripts/gpu_lock.sh run evs-v030 -- scripts/evs_v030_check.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export VLLM_IMAGE=vllm/vllm-openai:v0.30.0
OUT=results/evs_v030; mkdir -p "$OUT/arms"

serve () {  # base rate
  local arm="$OUT/arms/$1-evs$2.yaml"
  sed -E "s/^(server_args:.*)$/\1 --video-pruning-rate $2/" "arms/$1.yaml" > "$arm"
  docker/run_server.sh stop >/dev/null
  GPU_UTIL=0.80 docker/run_server.sh start "$arm" || return 1
  docker logs -f vlm-server > "$OUT/server-$1-evs$2.log" 2>&1 &
}
alive () { echo "  server alive: $(docker ps -q -f name=vlm-server | wc -l)"; }

if [ "${CONTROL:-0}" != 1 ]; then
echo "### 1. Qwen3-VL-4B, EVS 0.75, $VLLM_IMAGE $(date +%H:%M)"
if serve E_q3vl_4b 0.75; then
  docker/run_harness.sh python3 tools/evs_check.py --rate 0.75 --kinds img2,vid2,vid8 \
    --out "$OUT/q3vl4b-0.75.jsonl"
  alive
fi

echo "### 2. Qwen2.5-VL-7B, EVS 0.75, 8 frames at concurrency 32 $(date +%H:%M)"
if serve V_vllm_video 0.75; then
  docker/run_harness.sh python3 experiments/e8_vlm.py --arm arms/V_vllm_video.yaml \
    --inputs vid8 --evs 0.75 --limit-clips 3 --concurrency 32 --out "$OUT"
  alive
fi
docker/run_server.sh stop
echo "### v0.30.0 check COMPLETE $(date +%H:%M)"
fi

# Control for 2 (run with CONTROL=1): the same load with EVS off. If this survives, the OOM
# is EVS's, not 8-frame video's.
if [ "${CONTROL:-0}" = 1 ]; then
  echo "### 3. control: Qwen2.5-VL-7B, EVS off, 8 frames at concurrency 32 $(date +%H:%M)"
  docker/run_server.sh stop >/dev/null
  if GPU_UTIL=0.80 docker/run_server.sh start arms/V_vllm_video.yaml; then
    docker logs -f vlm-server > "$OUT/server-V_vllm_video-evs0.log" 2>&1 &
    docker/run_harness.sh python3 experiments/e8_vlm.py --arm arms/V_vllm_video.yaml \
      --inputs vid8 --limit-clips 3 --concurrency 32 --out "$OUT"
    alive
  fi
  docker/run_server.sh stop
  echo "### control COMPLETE $(date +%H:%M)"
fi
