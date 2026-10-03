#!/usr/bin/env bash
# Exploratory EVS feasibility check (tools/evs_check.py): each model at pruning rate 0
# and 0.75. Under the GPU lock:
#   scripts/gpu_lock.sh run evs-check -- scripts/evs_check.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/evs_check; mkdir -p "$OUT/arms"

for base in V_vllm_video E_q3vl_4b; do
  for rate in 0 0.75; do
    arm="$OUT/arms/$base-evs$rate.yaml"
    # the arm unchanged, plus the pruning flag (a later flag wins on the vLLM CLI)
    sed -E "s/^(server_args:.*)$/\1 --video-pruning-rate $rate/" "arms/$base.yaml" > "$arm"
    grep -q -- "--video-pruning-rate $rate" "$arm" || { echo "no server_args line in $base" >&2; exit 2; }
    echo "### $base rate $rate $(date +%H:%M)"
    docker/run_server.sh stop >/dev/null
    if GPU_UTIL=0.80 docker/run_server.sh start "$arm"; then
      docker/run_harness.sh python3 tools/evs_check.py --rate "$rate" --out "$OUT/$base-$rate.jsonl"
    else
      echo "### server failed to start"
    fi
    docker logs vlm-server > "$OUT/server-$base-$rate.log" 2>&1 || true
  done
done
docker/run_server.sh stop
echo "### EVS check COMPLETE $(date +%H:%M)"
