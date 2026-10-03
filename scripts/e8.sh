#!/usr/bin/env bash
# E8 (PLAN.md 14): the window sent as two images, one video, or one EVS-pruned video.
#
#   PASS=recog scripts/gpu_lock.sh run e8-recog -- scripts/e8.sh   # recognition, every arm
#   PASS=live  scripts/gpu_lock.sh run e8-live  -- scripts/e8.sh   # cameras per 3090
#
# EVS is a server flag, so each pruning rate is its own server start: the arm YAML
# unchanged plus --video-pruning-rate (a later flag wins on the vLLM CLI). Server
# logs are kept per start; the Qwen3-VL EVS crash (PLAN.md 14) only shows up there.
set -uo pipefail
cd "$(dirname "$0")/.."
PASS="${PASS:-recog}"
CAMS="${CAMS:-6,8,10,12,14,16,18,20,22,24}"
OUT=results/e8; mkdir -p "$OUT/arms"

serve () {  # arm evs-rate
  local arm="arms/$1.yaml"
  if [ "$2" != 0 ]; then
    arm="$OUT/arms/$1-evs$2.yaml"
    sed -E "s/^(server_args:.*)$/\1 --video-pruning-rate $2/" "arms/$1.yaml" > "$arm"
    grep -q -- "--video-pruning-rate $2" "$arm" || { echo "no server_args line in $1" >&2; return 1; }
  fi
  docker/run_server.sh stop >/dev/null
  GPU_UTIL=0.80 docker/run_server.sh start "$arm"
}

keep_log () { docker logs vlm-server > "$OUT/server-$PASS-$1.log" 2>&1 || true; }

recog () {  # arm evs-rate inputs
  echo "### recog $1 evs $2 [$3] $(date +%H:%M)"
  if serve "$1" "$2" && docker/run_harness.sh python3 experiments/e8_vlm.py \
       --arm "arms/$1.yaml" --inputs "$3" --evs "$2"; then
    echo "### $1 evs $2 DONE $(date +%H:%M)"
  else
    echo "### $1 evs $2 FAILED $(date +%H:%M)"
  fi
  keep_log "$1-evs$2"
}

live () {  # arm evs-rate label input
  echo "--- live $1 $3 $(date +%H:%M)"
  ALLOW_CONCURRENT=1 DS_TIMEOUT=5400 docker/run_ds.sh python3 experiments/e7c.py live \
    --arm-file "arms/$1.yaml" --gate track-motion --arm full --input "$4" \
    --cams "$CAMS" --search --discard-first --out "$OUT/$1/$3" \
    && echo "--- $1 $3 DONE $(date +%H:%M)" || echo "--- $1 $3 FAILED $(date +%H:%M)"
  sleep 10
}

case "$PASS" in
  recog)
    recog V_vllm_video 0 img2,vid2,vid8
    recog V_vllm_video 0.5 vid8
    recog V_vllm_video 0.75 vid8
    recog E_q3vl_4b 0 img2,vid2,vid8
    recog E_q3vl_8b 0 img2,vid2,vid8
    ;;
  live)
    if serve V_vllm_video 0; then
      live V_vllm_video 0 img2 img2; live V_vllm_video 0 vid2 vid2; live V_vllm_video 0 vid8 vid8
    fi
    keep_log V_vllm_video-evs0
    if serve V_vllm_video 0.75; then live V_vllm_video 0.75 vid8-e75 vid8; fi
    keep_log V_vllm_video-evs0.75
    if serve E_q3vl_4b 0; then live E_q3vl_4b 0 img2 img2; live E_q3vl_4b 0 vid2 vid2; fi
    keep_log E_q3vl_4b-evs0
    ;;
esac
docker/run_server.sh stop
echo "### E8 $PASS COMPLETE $(date +%H:%M)"
